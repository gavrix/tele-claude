import asyncio
import base64
import contextlib
from typing import Any, Dict, Optional, Tuple

from .cdp import CDPSession
from .errors import ElementNotFoundError, EvaluationError, NavigationError, TimeoutError


def _ax_value(node_value: Any) -> Optional[str]:
    if isinstance(node_value, dict):
        return node_value.get("value")
    return node_value


class Page:
    """Controls a single browser tab."""

    def __init__(
        self,
        session: CDPSession,
        session_id: str,
        target_id: str,
        browser: Any,
        default_timeout: float = 30.0,
    ):
        self._session = session
        self._session_id = session_id
        self._target_id = target_id
        self._browser = browser
        self._default_timeout = default_timeout
        self._url = "about:blank"
        self._initialized = False
        self._closed = False

    async def _initialize(self) -> None:
        if self._initialized:
            return
        await self._session.send("Page.enable", session_id=self._session_id)
        await self._session.send("Runtime.enable", session_id=self._session_id)
        await self._session.send("DOM.enable", session_id=self._session_id)
        self._session.on("Page.frameNavigated", self._on_frame_navigated, session_id=self._session_id)
        self._session.on("Page.navigatedWithinDocument", self._on_navigated_within_document, session_id=self._session_id)
        self._initialized = True

    def _mark_closed(self) -> None:
        self._closed = True

    @property
    def url(self) -> str:
        return self._url

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: Optional[float] = None) -> None:
        await self._ensure_open()
        event_method = "Page.domContentEventFired" if wait_until == "domcontentloaded" else "Page.loadEventFired"
        wait_task = asyncio.create_task(
            self._session.wait_for_event(event_method, session_id=self._session_id, timeout=timeout or self._default_timeout)
        )
        try:
            result = await self._session.send(
                "Page.navigate", {"url": url}, session_id=self._session_id, timeout=timeout or self._default_timeout
            )
            if result.get("errorText"):
                raise NavigationError(result["errorText"])
            await wait_task
        except TimeoutError as exc:
            raise NavigationError(f"Navigation to {url} timed out") from exc
        finally:
            if not wait_task.done():
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wait_task

    async def reload(self, wait_until: str = "domcontentloaded", timeout: Optional[float] = None) -> None:
        await self._ensure_open()
        event_method = "Page.domContentEventFired" if wait_until == "domcontentloaded" else "Page.loadEventFired"
        wait_task = asyncio.create_task(
            self._session.wait_for_event(event_method, session_id=self._session_id, timeout=timeout or self._default_timeout)
        )
        await self._session.send("Page.reload", {}, session_id=self._session_id, timeout=timeout or self._default_timeout)
        try:
            await wait_task
        except TimeoutError as exc:
            raise NavigationError("Reload timed out") from exc
        finally:
            if not wait_task.done():
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wait_task

    async def go_back(self, timeout: Optional[float] = None) -> bool:
        return await self._navigate_history(delta=-1, timeout=timeout)

    async def go_forward(self, timeout: Optional[float] = None) -> bool:
        return await self._navigate_history(delta=1, timeout=timeout)

    async def wait_for_navigation(self, wait_until: str = "load", timeout: Optional[float] = None) -> None:
        event_method = "Page.loadEventFired" if wait_until == "load" else "Page.domContentEventFired"
        try:
            await self._session.wait_for_event(event_method, session_id=self._session_id, timeout=timeout or self._default_timeout)
        except TimeoutError as exc:
            raise NavigationError("Navigation wait timed out") from exc

    async def title(self, timeout: Optional[float] = None) -> str:
        value = await self.evaluate("document.title", timeout=timeout)
        return value or ""

    async def content(self, timeout: Optional[float] = None) -> str:
        html = await self.evaluate("document.documentElement.outerHTML", timeout=timeout)
        return html or ""

    async def evaluate(self, expression: str, timeout: Optional[float] = None) -> Any:
        await self._ensure_open()
        result = await self._session.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )
        if result.get("exceptionDetails"):
            raise EvaluationError(result["exceptionDetails"].get("text", "JavaScript evaluation failed"))
        remote = result.get("result", {})
        if "value" in remote:
            return remote["value"]
        return remote

    async def screenshot(self, path: Optional[str] = None, full_page: bool = False, timeout: Optional[float] = None) -> Optional[bytes]:
        await self._ensure_open()
        params: Dict[str, Any] = {"format": "png", "captureBeyondViewport": True}
        if full_page:
            metrics = await self._session.send("Page.getLayoutMetrics", session_id=self._session_id, timeout=timeout or self._default_timeout)
            content_size = metrics.get("contentSize", {})
            width = max(1, int(content_size.get("width", 0)))
            height = max(1, int(content_size.get("height", 0)))
            params["clip"] = {"x": 0, "y": 0, "width": width, "height": height, "scale": 1}
        result = await self._session.send(
            "Page.captureScreenshot", params, session_id=self._session_id, timeout=timeout or self._default_timeout
        )
        data = base64.b64decode(result["data"])
        if path:
            with open(path, "wb") as f:
                f.write(data)
            return None
        return data

    async def accessibility_tree(self, timeout: Optional[float] = None) -> str:
        nodes = await self._session.send(
            "Accessibility.getFullAXTree",
            {"maxDepth": 5},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )
        return self._format_ax_tree(nodes.get("nodes", []))

    async def click(self, role: str, name: Optional[str] = None, index: int = 0, timeout: Optional[float] = None) -> None:
        backend_id, _snippet = await self._find_node_backend_id(role, name, index=index, timeout=timeout)
        await self._scroll_into_view(backend_id, timeout)
        x, y = await self._element_center(backend_id, timeout)
        for event_type in ("mousePressed", "mouseReleased"):
            await self._session.send(
                "Input.dispatchMouseEvent",
                {"type": event_type, "x": x, "y": y, "button": "left", "clickCount": 1},
                session_id=self._session_id,
                timeout=timeout or self._default_timeout,
            )
        return None

    async def type(
        self, role: str, name: Optional[str], text: str, clear: bool = True, timeout: Optional[float] = None
    ) -> None:
        backend_id, _snippet = await self._find_node_backend_id(role, name, timeout=timeout)
        await self._scroll_into_view(backend_id, timeout)
        object_id = await self._resolve_object_id(backend_id, timeout)
        await self._session.send("DOM.focus", {"backendNodeId": backend_id}, session_id=self._session_id, timeout=timeout or self._default_timeout)
        if clear:
            await self._session.send(
                "Runtime.callFunctionOn",
                {"objectId": object_id, "functionDeclaration": "function() { if ('value' in this) { this.value = ''; } }"},
                session_id=self._session_id,
                timeout=timeout or self._default_timeout,
            )
        await self._session.send(
            "Input.insertText", {"text": text}, session_id=self._session_id, timeout=timeout or self._default_timeout
        )

    async def press(self, key: str, timeout: Optional[float] = None) -> None:
        await self._ensure_open()
        for event_type in ("keyDown", "keyUp"):
            await self._session.send(
                "Input.dispatchKeyEvent",
                {"type": event_type, "key": key, "text": key if len(key) == 1 else ""},
                session_id=self._session_id,
                timeout=timeout or self._default_timeout,
            )

    async def close(self) -> None:
        if self._closed:
            return
        await self._session.send("Target.closeTarget", {"targetId": self._target_id}, timeout=self._default_timeout)
        self._closed = True
        if self._browser and self._target_id in getattr(self._browser, "_pages", {}):
            self._browser._pages.pop(self._target_id, None)

    # Internal helpers
    def _on_frame_navigated(self, params: dict) -> None:
        frame = params.get("frame", {})
        if frame and not frame.get("parentId"):
            self._url = frame.get("url", self._url)

    def _on_navigated_within_document(self, params: dict) -> None:
        self._url = params.get("url", self._url)

    async def _ensure_open(self) -> None:
        if self._closed:
            raise NavigationError("Page is closed")

    async def _navigate_history(self, delta: int, timeout: Optional[float]) -> bool:
        history = await self._session.send("Page.getNavigationHistory", session_id=self._session_id, timeout=timeout or self._default_timeout)
        entries = history.get("entries", [])
        current = history.get("currentIndex", 0)
        target_index = current + delta
        if target_index < 0 or target_index >= len(entries):
            return False
        await self._session.send(
            "Page.navigateToHistoryEntry",
            {"entryId": entries[target_index]["id"]},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )
        await self.wait_for_navigation(timeout=timeout or self._default_timeout)
        return True

    async def _find_node_backend_id(
        self, role: str, name: Optional[str], index: int = 0, timeout: Optional[float] = None
    ) -> Tuple[int, str]:
        # queryAXTree requires a starting node; use getFullAXTree and filter instead
        ax_result = await self._session.send(
            "Accessibility.getFullAXTree",
            {},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )
        nodes = ax_result
        candidates = []
        for node in nodes.get("nodes", []):
            node_role = _ax_value(node.get("role"))
            node_name = _ax_value(node.get("name"))
            if role and node_role != role:
                continue
            if name and node_name != name:
                continue
            backend_id = node.get("backendDOMNodeId")
            if backend_id:
                candidates.append((backend_id, node))

        if index < len(candidates):
            backend_id, _node = candidates[index]
            snippet = self._format_ax_tree(nodes.get("nodes", []))
            return backend_id, snippet

        snippet = self._format_ax_tree(nodes.get("nodes", []))
        raise ElementNotFoundError(role, name, snippet)

    async def _scroll_into_view(self, backend_node_id: int, timeout: Optional[float]) -> None:
        await self._session.send(
            "DOM.scrollIntoViewIfNeeded",
            {"backendNodeId": backend_node_id},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )

    async def _element_center(self, backend_node_id: int, timeout: Optional[float]) -> Tuple[float, float]:
        model = await self._session.send(
            "DOM.getBoxModel",
            {"backendNodeId": backend_node_id},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )
        box = model.get("model", {})
        quad = box.get("content") or box.get("padding") or box.get("border") or box.get("margin")
        if not quad:
            raise ElementNotFoundError("unknown", None, "Element box could not be determined")
        xs = quad[0::2]
        ys = quad[1::2]
        x = sum(xs) / len(xs)
        y = sum(ys) / len(ys)
        return x, y

    async def _resolve_object_id(self, backend_node_id: int, timeout: Optional[float]) -> str:
        result = await self._session.send(
            "DOM.resolveNode",
            {"backendNodeId": backend_node_id},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )
        obj = result.get("object", {})
        return obj["objectId"]

    def _format_ax_tree(self, nodes: list[dict], max_lines: int = 60) -> str:
        if not nodes:
            return "(empty accessibility tree)"
        node_map = {n["nodeId"]: n for n in nodes if "nodeId" in n}
        parents: dict[str, str] = {}
        for node in nodes:
            node_id = node.get("nodeId")
            if node_id is not None:
                for child_id in node.get("childIds", []):
                    parents[child_id] = node_id
        roots = [nid for nid in node_map if nid not in parents]
        lines: list[str] = []

        def walk(node_id: str, depth: int) -> None:
            if len(lines) >= max_lines:
                return
            node = node_map.get(node_id)
            if not node:
                return
            role = _ax_value(node.get("role")) or "node"
            name = _ax_value(node.get("name"))
            line = f"{'  '*depth}- {role}"
            if name:
                line += f' "{name}"'
            lines.append(line)
            for child in node.get("childIds", []):
                if child in node_map:
                    walk(child, depth + 1)

        for root in roots:
            walk(root, 0)
            if len(lines) >= max_lines:
                break
        return "\n".join(lines)
