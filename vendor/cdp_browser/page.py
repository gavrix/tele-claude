import asyncio
import base64
import contextlib
from typing import Any, Dict, List, Optional, Tuple

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
        params: Dict[str, Any] = {"format": "png", "captureBeyondViewport": full_page}
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

    async def find_by_text(
        self,
        text: str,
        exact: bool = False,
        max_matches: int = 10,
        ancestors: int = 3,
        descendants: int = 2,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Find elements by text content, return with backend node IDs and bounds.

        Args:
            text: Text to search for
            exact: If True, exact match. If False, case-insensitive contains
            max_matches: Maximum number of matches to return
            ancestors: How many ancestor levels to include in snippet
            descendants: How many descendant levels to include in snippet
            timeout: Operation timeout

        Returns:
            Dict with matches, each containing backend_node_id, bounds, snippet, clickable_ancestor
        """
        await self._ensure_open()

        # Escape text for JS string
        escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

        # JavaScript to find elements by text using TreeWalker
        find_js = f"""
        (() => {{
            const searchText = '{escaped_text}';
            const exact = {str(exact).lower()};
            const maxMatches = {max_matches};

            const CLICKABLE_SELECTOR = [
                'button',
                'a[href]',
                'input[type="button"]',
                'input[type="submit"]',
                '[role="button"]',
                '[role="link"]',
                '[role="menuitem"]',
                '[role="tab"]',
                '[role="option"]',
                '[role="checkbox"]',
                '[role="radio"]',
                '[role="switch"]',
                '[tabindex]:not([tabindex="-1"])'
            ].join(',');

            function isVisible(el) {{
                if (!el || !el.getBoundingClientRect) return false;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return false;
                const style = getComputedStyle(el);
                if (style.visibility === 'hidden' || style.display === 'none') return false;
                return true;
            }}

            function isClickable(el) {{
                if (!el || !el.matches) return false;
                try {{
                    if (el.matches(CLICKABLE_SELECTOR)) {{
                        const style = getComputedStyle(el);
                        return style.pointerEvents !== 'none'
                            && style.visibility !== 'hidden'
                            && !el.disabled;
                    }}
                }} catch (e) {{}}
                return false;
            }}

            function findClickableAncestor(el) {{
                let current = el;
                while (current && current !== document.body) {{
                    if (isClickable(current)) {{
                        return current;
                    }}
                    current = current.parentElement;
                }}
                return null;
            }}

            function getElementInfo(el) {{
                const rect = el.getBoundingClientRect();
                return {{
                    tag: el.tagName.toLowerCase(),
                    id: el.id || null,
                    className: el.className ? (typeof el.className === 'string' ? el.className.split(' ')[0] : null) : null,
                    role: el.getAttribute('role'),
                    textContent: (el.textContent || '').trim().substring(0, 100)
                }};
            }}

            // Use TreeWalker for efficient text node search
            const matches = [];
            const seenElements = new Set();

            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                {{
                    acceptNode: (node) => {{
                        const nodeText = node.textContent || '';
                        if (!nodeText.trim()) return NodeFilter.FILTER_REJECT;

                        const matches = exact
                            ? nodeText.trim() === searchText
                            : nodeText.toLowerCase().includes(searchText.toLowerCase());

                        return matches ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                    }}
                }}
            );

            while (walker.nextNode() && matches.length < maxMatches) {{
                const textNode = walker.currentNode;
                const parentEl = textNode.parentElement;

                if (!parentEl || seenElements.has(parentEl)) continue;
                if (!isVisible(parentEl)) continue;

                seenElements.add(parentEl);

                // Find matched substring
                const nodeText = textNode.textContent || '';
                let matchedText = searchText;
                if (!exact) {{
                    const lowerText = nodeText.toLowerCase();
                    const idx = lowerText.indexOf(searchText.toLowerCase());
                    if (idx >= 0) {{
                        matchedText = nodeText.substring(idx, idx + searchText.length);
                    }}
                }}

                const rect = parentEl.getBoundingClientRect();
                const clickableAncestor = findClickableAncestor(parentEl);

                matches.push({{
                    matchedText,
                    element: getElementInfo(parentEl),
                    bounds: {{
                        x: Math.round(rect.x),
                        y: Math.round(rect.y),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    }},
                    clickableAncestor: clickableAncestor ? getElementInfo(clickableAncestor) : null,
                    // Store references for later backend ID resolution
                    _elementRef: parentEl,
                    _clickableRef: clickableAncestor
                }});
            }}

            return {{
                matches,
                totalFound: matches.length,
                viewport: {{
                    width: window.innerWidth,
                    height: window.innerHeight
                }},
                devicePixelRatio: window.devicePixelRatio || 1
            }};
        }})()
        """

        # Execute the search
        result = await self._session.send(
            "Runtime.evaluate",
            {
                "expression": find_js,
                "returnByValue": False,  # Need object references
                "awaitPromise": False,
            },
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )

        if result.get("exceptionDetails"):
            raise EvaluationError(f"Find failed: {result['exceptionDetails'].get('text', 'unknown error')}")

        # Get the result object ID
        result_object_id = result.get("result", {}).get("objectId")
        if not result_object_id:
            # Fallback: result was returned by value
            return {"matches": [], "totalFound": 0, "search_text": text, "search_mode": "exact" if exact else "contains"}

        # Get properties from the result
        props_result = await self._session.send(
            "Runtime.getProperties",
            {"objectId": result_object_id, "ownProperties": True},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )

        # Parse the result
        result_data: Dict[str, Any] = {"matches": [], "totalFound": 0}
        for prop in props_result.get("result", []):
            name = prop.get("name")
            value = prop.get("value", {})
            if name == "totalFound":
                result_data["totalFound"] = value.get("value", 0)
            elif name == "viewport":
                vp_id = value.get("objectId")
                if vp_id:
                    vp_props = await self._session.send(
                        "Runtime.getProperties",
                        {"objectId": vp_id, "ownProperties": True},
                        session_id=self._session_id,
                        timeout=timeout or self._default_timeout,
                    )
                    result_data["viewport"] = {
                        p["name"]: p["value"]["value"]
                        for p in vp_props.get("result", [])
                        if p.get("value", {}).get("value") is not None
                    }
            elif name == "devicePixelRatio":
                result_data["devicePixelRatio"] = value.get("value", 1)
            elif name == "matches":
                matches_id = value.get("objectId")
                if matches_id:
                    result_data["matches"] = await self._process_find_matches(
                        matches_id, ancestors, descendants, timeout
                    )

        result_data["search_text"] = text
        result_data["search_mode"] = "exact" if exact else "contains"
        result_data["returned"] = len(result_data["matches"])

        return result_data

    async def _process_find_matches(
        self,
        matches_object_id: str,
        ancestors: int,
        descendants: int,
        timeout: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Process match array and resolve backend node IDs."""
        matches_props = await self._session.send(
            "Runtime.getProperties",
            {"objectId": matches_object_id, "ownProperties": True},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )

        processed_matches = []

        for prop in matches_props.get("result", []):
            if not prop.get("name", "").isdigit():
                continue

            match_id = prop.get("value", {}).get("objectId")
            if not match_id:
                continue

            match_data = await self._extract_match_data(match_id, ancestors, descendants, timeout)
            if match_data:
                match_data["index"] = int(prop["name"])
                processed_matches.append(match_data)

        # Sort by index
        processed_matches.sort(key=lambda m: m["index"])
        return processed_matches

    async def _extract_match_data(
        self,
        match_object_id: str,
        ancestors: int,
        descendants: int,
        timeout: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Extract data from a single match object."""
        props = await self._session.send(
            "Runtime.getProperties",
            {"objectId": match_object_id, "ownProperties": True},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )

        match_data: Dict[str, Any] = {}
        element_ref_id = None
        clickable_ref_id = None

        for prop in props.get("result", []):
            name = prop.get("name")
            value = prop.get("value", {})

            if name == "matchedText":
                match_data["matched_text"] = value.get("value", "")
            elif name == "bounds":
                bounds_id = value.get("objectId")
                if bounds_id:
                    bounds_props = await self._session.send(
                        "Runtime.getProperties",
                        {"objectId": bounds_id, "ownProperties": True},
                        session_id=self._session_id,
                        timeout=timeout or self._default_timeout,
                    )
                    match_data["bounds"] = {
                        p["name"]: p["value"]["value"]
                        for p in bounds_props.get("result", [])
                        if p.get("value", {}).get("value") is not None
                    }
            elif name == "element":
                el_id = value.get("objectId")
                if el_id:
                    match_data["element"] = await self._extract_element_info(el_id, timeout)
            elif name == "clickableAncestor":
                if value.get("type") != "null" and value.get("objectId"):
                    ca_id = value.get("objectId")
                    match_data["clickable_ancestor"] = await self._extract_element_info(ca_id, timeout)
                else:
                    match_data["clickable_ancestor"] = None
            elif name == "_elementRef":
                element_ref_id = value.get("objectId")
            elif name == "_clickableRef":
                if value.get("type") != "null":
                    clickable_ref_id = value.get("objectId")

        # Resolve backend node IDs
        if element_ref_id:
            try:
                node_desc = await self._session.send(
                    "DOM.describeNode",
                    {"objectId": element_ref_id},
                    session_id=self._session_id,
                    timeout=timeout or self._default_timeout,
                )
                backend_id = node_desc.get("node", {}).get("backendNodeId")
                if backend_id and "element" in match_data:
                    match_data["element"]["backend_node_id"] = backend_id
            except Exception:
                pass

        if clickable_ref_id:
            try:
                node_desc = await self._session.send(
                    "DOM.describeNode",
                    {"objectId": clickable_ref_id},
                    session_id=self._session_id,
                    timeout=timeout or self._default_timeout,
                )
                backend_id = node_desc.get("node", {}).get("backendNodeId")
                clickable_ancestor = match_data.get("clickable_ancestor")
                if backend_id and clickable_ancestor is not None:
                    clickable_ancestor["backend_node_id"] = backend_id
            except Exception:
                pass

        # Build snippet
        if element_ref_id:
            match_data["snippet"] = await self._build_snippet(
                element_ref_id, ancestors, descendants, timeout
            )

        return match_data if match_data else None

    async def _extract_element_info(
        self,
        object_id: str,
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Extract element info from object ID."""
        props = await self._session.send(
            "Runtime.getProperties",
            {"objectId": object_id, "ownProperties": True},
            session_id=self._session_id,
            timeout=timeout or self._default_timeout,
        )

        info = {}
        for prop in props.get("result", []):
            name = prop.get("name")
            value = prop.get("value", {}).get("value")
            if name in ("tag", "id", "className", "role", "textContent") and value is not None:
                info[name] = value
        return info

    async def _build_snippet(
        self,
        element_object_id: str,
        ancestors: int,
        descendants: int,
        timeout: Optional[float],
    ) -> str:
        """Build a compact snippet showing element context."""
        snippet_js = f"""
        (function(el) {{
            const ancestors = {ancestors};
            const descendants = {descendants};
            const lines = [];

            function getNodeLine(node, depth, isMatch) {{
                if (!node || !node.tagName) return null;
                const tag = node.tagName.toLowerCase();
                const id = node.id ? '#' + node.id : '';
                const cls = node.className && typeof node.className === 'string'
                    ? '.' + node.className.split(' ')[0]
                    : '';
                const role = node.getAttribute && node.getAttribute('role')
                    ? '[role=' + node.getAttribute('role') + ']'
                    : '';
                const indent = '  '.repeat(depth);
                let line = indent + '- ' + tag + id + cls + role;
                if (isMatch) line += ' <-- match';
                return line;
            }}

            function walkDescendants(node, depth, maxDepth) {{
                if (depth > maxDepth || !node) return;
                for (const child of (node.children || [])) {{
                    const line = getNodeLine(child, depth, false);
                    if (line) lines.push(line);
                    walkDescendants(child, depth + 1, maxDepth);
                }}
            }}

            // Collect ancestors
            const ancestorChain = [];
            let current = el;
            for (let i = 0; i < ancestors && current && current !== document.body; i++) {{
                if (current.parentElement) {{
                    ancestorChain.unshift(current.parentElement);
                    current = current.parentElement;
                }} else {{
                    break;
                }}
            }}

            // Build snippet
            let depth = 0;
            for (const ancestor of ancestorChain) {{
                const line = getNodeLine(ancestor, depth, false);
                if (line) lines.push(line);
                depth++;
            }}

            // The matched element
            const matchLine = getNodeLine(el, depth, true);
            if (matchLine) lines.push(matchLine);

            // Descendants
            walkDescendants(el, depth + 1, depth + descendants);

            return lines.join('\\n').substring(0, 400);
        }})(this)
        """

        try:
            result = await self._session.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": element_object_id,
                    "functionDeclaration": snippet_js,
                    "returnByValue": True,
                },
                session_id=self._session_id,
                timeout=timeout or self._default_timeout,
            )
            return result.get("result", {}).get("value", "(snippet unavailable)")
        except Exception:
            return "(snippet unavailable)"

    async def click_by_node_id(
        self,
        backend_node_id: int,
        timeout: Optional[float] = None,
    ) -> None:
        """Click element by backend node ID.

        Args:
            backend_node_id: The backend DOM node ID to click
            timeout: Operation timeout
        """
        await self._ensure_open()

        # Verify node still exists
        try:
            await self._session.send(
                "DOM.describeNode",
                {"backendNodeId": backend_node_id},
                session_id=self._session_id,
                timeout=timeout or self._default_timeout,
            )
        except Exception as e:
            raise ElementNotFoundError(
                "node",
                str(backend_node_id),
                f"Node with backend_node_id={backend_node_id} no longer exists. Re-run browser_find to get fresh IDs."
            ) from e

        # Scroll into view and click
        await self._scroll_into_view(backend_node_id, timeout)
        x, y = await self._element_center(backend_node_id, timeout)

        for event_type in ("mousePressed", "mouseReleased"):
            await self._session.send(
                "Input.dispatchMouseEvent",
                {"type": event_type, "x": x, "y": y, "button": "left", "clickCount": 1},
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

    def _format_ax_tree(self, nodes: list[dict], max_lines: int = 300) -> str:
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
