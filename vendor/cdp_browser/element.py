from typing import Optional

from .page import Page


class Element:
    """Represents a DOM element resolved from the accessibility tree."""

    def __init__(self, page: Page, backend_node_id: int, role: Optional[str] = None, name: Optional[str] = None):
        self._page = page
        self._backend_node_id = backend_node_id
        self._role = role
        self._name = name

    @property
    def role(self) -> Optional[str]:
        return self._role

    @property
    def name(self) -> Optional[str]:
        return self._name

    async def click(self, timeout: Optional[float] = None) -> None:
        await self._page._scroll_into_view(self._backend_node_id, timeout)
        x, y = await self._page._element_center(self._backend_node_id, timeout)
        for event_type in ("mousePressed", "mouseReleased"):
            await self._page._session.send(
                "Input.dispatchMouseEvent",
                {"type": event_type, "x": x, "y": y, "button": "left", "clickCount": 1},
                session_id=self._page._session_id,
                timeout=timeout or self._page._default_timeout,
            )

    async def type(self, text: str, clear: bool = True, timeout: Optional[float] = None) -> None:
        object_id = await self._page._resolve_object_id(self._backend_node_id, timeout)
        await self._page._session.send(
            "DOM.focus",
            {"backendNodeId": self._backend_node_id},
            session_id=self._page._session_id,
            timeout=timeout or self._page._default_timeout,
        )
        if clear:
            await self._page._session.send(
                "Runtime.callFunctionOn",
                {"objectId": object_id, "functionDeclaration": "function() { if ('value' in this) { this.value = ''; } }"},
                session_id=self._page._session_id,
                timeout=timeout or self._page._default_timeout,
            )
        await self._page._session.send(
            "Input.insertText", {"text": text}, session_id=self._page._session_id, timeout=timeout or self._page._default_timeout
        )

    async def focus(self, timeout: Optional[float] = None) -> None:
        await self._page._session.send(
            "DOM.focus",
            {"backendNodeId": self._backend_node_id},
            session_id=self._page._session_id,
            timeout=timeout or self._page._default_timeout,
        )

    async def scroll_into_view(self, timeout: Optional[float] = None) -> None:
        await self._page._scroll_into_view(self._backend_node_id, timeout)
