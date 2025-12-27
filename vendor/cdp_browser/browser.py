import asyncio
import json
from typing import Dict, List, Optional
from urllib.parse import urljoin
from urllib.request import urlopen

from .cdp import CDPSession
from .errors import ConnectionError
from .page import Page


class Browser:
    """Manage CDP connection and tab lifecycle."""

    def __init__(self, cdp_endpoint: str, default_timeout: float = 30.0):
        self._endpoint = cdp_endpoint.rstrip("/")
        self._default_timeout = default_timeout
        self._session: Optional[CDPSession] = None
        self._pages: Dict[str, Page] = {}
        self._owned_targets: set[str] = set()
        self._connected = False

    async def __aenter__(self) -> "Browser":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> "Browser":
        if self._connected:
            return self
        ws_url = await self._resolve_websocket_url()
        self._session = await CDPSession.connect(ws_url, default_timeout=self._default_timeout)
        self._session.on("Target.detachedFromTarget", self._on_target_detached)
        await self._session.send("Target.setDiscoverTargets", {"discover": True})
        self._connected = True
        return self

    async def close(self) -> None:
        if not self._connected:
            return
        # Close only owned targets
        for page in list(self._pages.values()):
            try:
                await page.close()
            except Exception:
                continue
        self._pages.clear()
        self._owned_targets.clear()
        if self._session:
            await self._session.close()
        self._connected = False

    async def new_page(self, url: str = "about:blank", background: bool = True) -> Page:
        await self._ensure_connected()
        assert self._session
        result = await self._session.send(
            "Target.createTarget",
            {"url": url, "background": background},
            timeout=self._default_timeout,
        )
        target_id = result["targetId"]
        self._owned_targets.add(target_id)
        attach = await self._session.send(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            timeout=self._default_timeout,
        )
        session_id = attach["sessionId"]
        page = Page(
            session=self._session,
            session_id=session_id,
            target_id=target_id,
            browser=self,
            default_timeout=self._default_timeout,
        )
        await page._initialize()
        self._pages[target_id] = page
        return page

    @property
    def pages(self) -> List[Page]:
        return list(self._pages.values())

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def _resolve_websocket_url(self) -> str:
        if self._endpoint.startswith("ws://") or self._endpoint.startswith("wss://"):
            return self._endpoint

        version_url = self._endpoint
        if not version_url.endswith("/json/version"):
            version_url = urljoin(version_url + "/", "json/version")

        loop = asyncio.get_event_loop()

        def _fetch() -> str:
            with urlopen(version_url) as resp:
                data = resp.read()
            payload = json.loads(data.decode("utf-8"))
            return payload.get("webSocketDebuggerUrl", "")

        ws_url = await loop.run_in_executor(None, _fetch)
        if not ws_url:
            raise ConnectionError(f"Could not resolve websocket debugger URL from {version_url}")
        return ws_url

    def _on_target_detached(self, params: dict) -> None:
        target_id = params.get("targetId")
        if target_id in self._pages:
            page = self._pages.pop(target_id)
            page._mark_closed()
