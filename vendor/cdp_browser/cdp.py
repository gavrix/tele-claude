import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets

from .errors import CDPError, ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class CDPSession:
    """Low-level CDP client handling message routing, timeouts, and events."""

    def __init__(self, websocket, default_timeout: float = 30.0, ping_interval: float = 20.0):
        self._ws = websocket
        self._default_timeout = default_timeout
        self._ping_interval = ping_interval
        self._pending: Dict[int, asyncio.Future] = {}
        self._event_handlers: Dict[Optional[str], Dict[str, list[Callable[[Dict[str, Any]], Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._id = 0
        self._recv_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._closed = False

    @classmethod
    async def connect(cls, websocket_url: str, default_timeout: float = 30.0, ping_interval: float = 20.0) -> "CDPSession":
        try:
            ws = await websockets.connect(websocket_url, ping_interval=None, max_size=None)
        except Exception as exc:  # pragma: no cover - exercised in integration
            raise ConnectionError(f"Failed to connect to {websocket_url}") from exc

        session = cls(ws, default_timeout=default_timeout, ping_interval=ping_interval)
        session._recv_task = asyncio.create_task(session._recv_loop(), name="cdp-recv-loop")
        if ping_interval:
            session._ping_task = asyncio.create_task(session._ping_loop(), name="cdp-ping-loop")
        return session

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(ConnectionError("CDP session closed"))
        self._pending.clear()
        if self._ws:
            await self._ws.close()

    async def send(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if self._closed:
            raise ConnectionError("CDP session is closed")
        self._id += 1
        msg_id = self._id
        payload = {"id": msg_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        serialized = json.dumps(payload)
        await self._ws.send(serialized)

        use_timeout = timeout if timeout is not None else self._default_timeout
        try:
            return await asyncio.wait_for(fut, timeout=use_timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"CDP method {method} timed out after {use_timeout}s") from exc

    def on(self, method: str, handler: Callable[[Dict[str, Any]], Any], session_id: Optional[str] = None) -> None:
        """Register an event handler for a method on a given session."""
        self._event_handlers[session_id][method].append(handler)

    def off(self, method: str, handler: Callable[[Dict[str, Any]], Any], session_id: Optional[str] = None) -> None:
        handlers = self._event_handlers.get(session_id, {}).get(method, [])
        if handler in handlers:
            handlers.remove(handler)

    async def wait_for_event(
        self,
        method: str,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        def _handler(event: Dict[str, Any]) -> None:
            if predicate is None or predicate(event):
                if not fut.done():
                    fut.set_result(event)

        self.on(method, _handler, session_id=session_id)
        try:
            use_timeout = timeout if timeout is not None else self._default_timeout
            return await asyncio.wait_for(fut, timeout=use_timeout)
        finally:
            self.off(method, _handler, session_id=session_id)

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON CDP message: %s", raw)
                    continue
                await self._handle_message(message)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # pragma: no cover - integration only
            logger.warning("CDP recv loop failed: %s", exc)
            await self._fail_all(ConnectionError("CDP connection dropped"))
        finally:
            self._closed = True

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        if "id" in message:
            msg_id = message["id"]
            fut = self._pending.pop(msg_id, None)
            if not fut:
                return
            if "error" in message:
                err = message["error"]
                fut.set_exception(CDPError(err.get("message", "CDP command failed")))
            else:
                fut.set_result(message.get("result"))
            return

        # Event
        method = message.get("method")
        if method is None:
            return
        params = message.get("params", {})
        session_id = message.get("sessionId")
        for sid in {session_id, None}:
            for handler in list(self._event_handlers.get(sid, {}).get(method, [])):
                try:
                    result = handler(params)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("CDP event handler error for %s", method)

    async def _ping_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._ping_interval)
                try:
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=self._ping_interval / 2)
                except Exception:
                    logger.warning("CDP ping failed; closing session")
                    await self.close()
        except asyncio.CancelledError:
            return

    async def _fail_all(self, exc: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
