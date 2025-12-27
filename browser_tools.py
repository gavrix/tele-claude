"""
Browser automation tools for Claude Agent SDK.

Provides browser control with:
- Fast CDP connection via cdp-browser library (only creates ONE new tab, doesn't touch existing tabs)
- Persistent context for cookies/authentication via Playwright (non-CDP mode)
- Accessibility tree extraction for element targeting
- Screenshot capture for visual understanding
"""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

# Import vendored cdp-browser library
from vendor.cdp_browser import Browser as CDPBrowser, Page as CDPPage
from vendor.cdp_browser.errors import ElementNotFoundError as CDPElementNotFoundError
from playwright.async_api import async_playwright, Browser as PWBrowser, BrowserContext, Page as PWPage, Playwright

from claude_agent_sdk import create_sdk_mcp_server, tool

if TYPE_CHECKING:
    from session import ClaudeSession

# Configuration (imported from config.py at runtime to avoid circular imports)
ACCESSIBILITY_MAX_LENGTH = 50000  # Truncate if larger

# Global instances (reused across sessions)
_playwright: Optional[Playwright] = None
_playwright_lock = asyncio.Lock()
_cdp_browser: Optional[CDPBrowser] = None
_cdp_browser_lock = asyncio.Lock()


async def get_playwright() -> Playwright:
    """Get or create global playwright instance."""
    global _playwright
    async with _playwright_lock:
        if _playwright is None:
            _playwright = await async_playwright().start()
        return _playwright


async def get_cdp_browser(endpoint: str) -> CDPBrowser:
    """Get or create global CDP browser connection."""
    global _cdp_browser
    async with _cdp_browser_lock:
        if _cdp_browser is None:
            _cdp_browser = await CDPBrowser(endpoint).connect()
        return _cdp_browser


@dataclass
class BrowserSession:
    """Manages browser state for a Claude session."""
    # For CDP mode: uses cdp-browser
    cdp_page: Optional[CDPPage] = None
    # For non-CDP mode: uses Playwright
    pw_context: Optional[BrowserContext] = None
    pw_page: Optional[PWPage] = None
    user_data_dir: Optional[Path] = None
    last_activity: float = field(default_factory=time.time)

    @property
    def is_cdp(self) -> bool:
        return self.cdp_page is not None

    @property
    def page(self) -> Union[CDPPage, PWPage]:
        """Get the active page (either CDP or Playwright)."""
        if self.cdp_page:
            return self.cdp_page
        if self.pw_page:
            return self.pw_page
        raise RuntimeError("No page available")

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    async def close(self) -> None:
        """Close browser session."""
        try:
            if self.cdp_page:
                await self.cdp_page.close()
                self.cdp_page = None
            if self.pw_context:
                await self.pw_context.close()
                self.pw_context = None
                self.pw_page = None
        except Exception:
            pass


async def create_browser_session(thread_id: int, headless: bool = True, data_dir: Optional[Path] = None) -> BrowserSession:
    """Create new browser session with persistent context or CDP connection.

    For CDP mode: Uses cdp-browser library to create a SINGLE new tab in the
    user's Chrome without attaching to or interfering with existing tabs.
    This connects in <1 second regardless of how many tabs are open.

    Args:
        thread_id: Telegram thread ID for isolation
        headless: Run browser in headless mode (ignored for CDP)
        data_dir: Base directory for browser data persistence (ignored for CDP)
    """
    from config import BROWSER_CDP_ENDPOINT, BROWSER_DATA_DIR, BROWSER_HEADLESS

    if BROWSER_CDP_ENDPOINT:
        # Use cdp-browser for fast CDP connection
        browser = await get_cdp_browser(BROWSER_CDP_ENDPOINT)
        page = await browser.new_page()
        return BrowserSession(cdp_page=page)

    # Otherwise launch our own browser with Playwright persistent context
    pw = await get_playwright()

    if data_dir is None:
        data_dir = BROWSER_DATA_DIR
    if headless is None:
        headless = BROWSER_HEADLESS

    # Create user data dir for persistence (per telegram thread)
    user_data_dir = data_dir / f"session_{thread_id}"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Launch persistent context (preserves cookies, localStorage, etc.)
    context = await pw.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=headless,
        viewport={"width": 1280, "height": 720},
        args=["--force-renderer-accessibility"]
    )

    # Get the default page or create one
    page = context.pages[0] if context.pages else await context.new_page()

    return BrowserSession(
        pw_context=context,
        pw_page=page,
        user_data_dir=user_data_dir
    )


async def ensure_browser(session: "ClaudeSession") -> BrowserSession:
    """Ensure browser session exists and is healthy, creating/reconnecting if needed."""
    from config import BROWSER_HEADLESS

    # Check if existing session is still alive
    if session.browser_session is not None:
        try:
            # Quick health check - verify the page is responsive
            bs = session.browser_session
            if bs.is_cdp and bs.cdp_page:
                # For CDP, check the page is still open
                await bs.cdp_page.evaluate("1 + 1")
            elif bs.pw_page:
                # For Playwright, check context is alive
                _ = bs.pw_page.url
                await bs.pw_page.evaluate("1 + 1")
        except Exception:
            # Connection is dead, clean up and reconnect
            try:
                await session.browser_session.close()
            except Exception:
                pass
            session.browser_session = None

    if session.browser_session is None:
        session.browser_session = await create_browser_session(session.thread_id, headless=BROWSER_HEADLESS)

    session.browser_session.touch()
    return session.browser_session


def format_accessibility_tree(node: dict, indent: int = 0) -> str:
    """Format accessibility tree as readable text.

    Output format:
    [role] "name" = value
      [child_role] "child_name"
    """
    lines = []
    prefix = "  " * indent

    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    # Skip generic/none roles without meaningful content
    if role in ("none", "generic") and not name:
        # Still process children
        for child in node.get("children", []):
            child_text = format_accessibility_tree(child, indent)
            if child_text.strip():
                lines.append(child_text)
        return "\n".join(lines)

    # Format: [role] "name" = value
    line = f"{prefix}[{role}]"
    if name:
        # Truncate very long names
        display_name = name[:100] + "..." if len(name) > 100 else name
        line += f' "{display_name}"'
    if value:
        display_value = value[:100] + "..." if len(value) > 100 else value
        line += f" = {display_value}"

    # Add additional useful attributes
    attrs = []
    if node.get("focused"):
        attrs.append("focused")
    if node.get("disabled"):
        attrs.append("disabled")
    if node.get("checked") is not None:
        attrs.append(f"checked={node['checked']}")
    if attrs:
        line += f" ({', '.join(attrs)})"

    lines.append(line)

    # Recurse into children
    for child in node.get("children", []):
        child_text = format_accessibility_tree(child, indent + 1)
        if child_text.strip():
            lines.append(child_text)

    return "\n".join(lines)


async def get_accessibility_tree(browser_session: BrowserSession) -> str:
    """Extract accessibility tree from page.

    For CDP mode: Uses cdp-browser's accessibility_tree() method
    For Playwright: Uses aria_snapshot() on body
    """
    try:
        if browser_session.is_cdp and browser_session.cdp_page:
            # Use cdp-browser's accessibility tree
            tree = await browser_session.cdp_page.accessibility_tree()
            if len(tree) > ACCESSIBILITY_MAX_LENGTH:
                tree = tree[:ACCESSIBILITY_MAX_LENGTH] + "\n... (truncated)"
            return tree
        elif browser_session.pw_page:
            # Use Playwright's aria_snapshot
            aria_snapshot = await browser_session.pw_page.locator("body").aria_snapshot()
            if aria_snapshot:
                if len(aria_snapshot) > ACCESSIBILITY_MAX_LENGTH:
                    aria_snapshot = aria_snapshot[:ACCESSIBILITY_MAX_LENGTH] + "\n... (truncated)"
                return aria_snapshot
            return "(empty page)"
        return "(no page available)"
    except Exception as e:
        return f"(accessibility tree unavailable: {e})"


async def take_screenshot(browser_session: BrowserSession, session: "ClaudeSession") -> str:
    """Take screenshot and return temp file path."""
    import tempfile
    screenshot_path = Path(tempfile.gettempdir()) / f"browser_screenshot_{session.thread_id}_{int(time.time() * 1000)}.png"

    if browser_session.is_cdp and browser_session.cdp_page:
        await browser_session.cdp_page.screenshot(path=str(screenshot_path))
    elif browser_session.pw_page:
        await browser_session.pw_page.screenshot(path=str(screenshot_path))

    return str(screenshot_path)


async def get_page_state(browser_session: BrowserSession, session: "ClaudeSession") -> dict[str, str]:
    """Get current page state including accessibility tree and screenshot."""
    tree = await get_accessibility_tree(browser_session)
    screenshot_path = await take_screenshot(browser_session, session)

    if browser_session.is_cdp and browser_session.cdp_page:
        url = browser_session.cdp_page.url
        title = await browser_session.cdp_page.title()
    elif browser_session.pw_page:
        url = browser_session.pw_page.url
        title = await browser_session.pw_page.title()
    else:
        url = "unknown"
        title = "unknown"

    return {
        "accessibility_tree": tree,
        "screenshot_path": screenshot_path,
        "current_url": url,
        "page_title": title,
    }


def create_browser_mcp_server(session: "ClaudeSession"):
    """Create MCP server with browser tools bound to session.

    The server runs in-process and has access to the session's browser
    via closure.

    Args:
        session: The ClaudeSession to bind tools to

    Returns:
        McpSdkServerConfig ready to use with ClaudeAgentOptions.mcp_servers
    """

    @tool(
        "browser_navigate",
        "Navigate browser to a URL. Returns accessibility tree and screenshot path for visual inspection.",
        {
            "url": str,
        }
    )
    async def browser_navigate(args: dict[str, Any]) -> dict[str, Any]:
        """Navigate to a URL."""
        url = args.get("url", "")
        if not url:
            return {
                "content": [{"type": "text", "text": "Error: url is required"}],
                "is_error": True
            }

        try:
            browser = await ensure_browser(session)

            # Navigate with reasonable timeout
            if browser.is_cdp and browser.cdp_page:
                await browser.cdp_page.goto(url, timeout=30.0)
            elif browser.pw_page:
                await browser.pw_page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Brief wait for dynamic content
            await asyncio.sleep(1)

            state = await get_page_state(browser, session)

            result_text = (
                f"Navigated to: {state['current_url']}\n"
                f"Title: {state['page_title']}\n"
                f"Screenshot: {state['screenshot_path']}\n\n"
                f"Accessibility Tree:\n{state['accessibility_tree']}"
            )

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Navigation error: {str(e)}"}],
                "is_error": True
            }

    @tool(
        "browser_snapshot",
        "Get current page state (accessibility tree and screenshot) without taking any action. "
        "Use this to refresh your understanding of the page.",
        {}
    )
    async def browser_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        """Get current page state."""
        try:
            browser = await ensure_browser(session)
            state = await get_page_state(browser, session)

            result_text = (
                f"URL: {state['current_url']}\n"
                f"Title: {state['page_title']}\n"
                f"Screenshot: {state['screenshot_path']}\n\n"
                f"Accessibility Tree:\n{state['accessibility_tree']}"
            )

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Snapshot error: {str(e)}"}],
                "is_error": True
            }

    @tool(
        "browser_click",
        "Click an element identified by its accessibility role and name. "
        "Use role and name from the accessibility tree (e.g., role='button', name='Submit'). "
        "Use index if there are multiple matches (0-indexed).",
        {
            "role": str,
            "name": str,
            "index": int,  # Optional, defaults to 0
        }
    )
    async def browser_click(args: dict[str, Any]) -> dict[str, Any]:
        """Click an element by role and name."""
        role = args.get("role", "")
        name = args.get("name", "")
        index = args.get("index", 0)

        if not role:
            return {
                "content": [{"type": "text", "text": "Error: role is required"}],
                "is_error": True
            }

        try:
            browser = await ensure_browser(session)

            if browser.is_cdp and browser.cdp_page:
                # Use cdp-browser click
                try:
                    await browser.cdp_page.click(role, name=name if name else None, index=index, timeout=10.0)
                except CDPElementNotFoundError as e:
                    state = await get_page_state(browser, session)
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"No element found with role='{role}'" + (f" name='{name}'" if name else "") +
                                    f"\n\nCurrent page state:\nScreenshot: {state['screenshot_path']}\n\n"
                                    f"Accessibility Tree:\n{state['accessibility_tree']}"
                        }],
                        "is_error": True
                    }
            elif browser.pw_page:
                # Use Playwright locator
                if name:
                    locator = browser.pw_page.get_by_role(role, name=name)
                else:
                    locator = browser.pw_page.get_by_role(role)

                count = await locator.count()
                if count == 0:
                    state = await get_page_state(browser, session)
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"No element found with role='{role}'" + (f" name='{name}'" if name else "") +
                                    f"\n\nCurrent page state:\nScreenshot: {state['screenshot_path']}\n\n"
                                    f"Accessibility Tree:\n{state['accessibility_tree']}"
                        }],
                        "is_error": True
                    }

                if index >= count:
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"Index {index} out of range. Found {count} element(s) with role='{role}'" +
                                    (f" name='{name}'" if name else "")
                        }],
                        "is_error": True
                    }

                await locator.nth(index).click()
                await browser.pw_page.wait_for_load_state("domcontentloaded", timeout=5000)

            # Brief wait for any dynamic updates after click
            await asyncio.sleep(0.5)

            state = await get_page_state(browser, session)

            result_text = (
                f"Clicked: [{role}] \"{name}\"" + (f" (index {index})" if index > 0 else "") + "\n"
                f"URL: {state['current_url']}\n"
                f"Screenshot: {state['screenshot_path']}\n\n"
                f"Accessibility Tree:\n{state['accessibility_tree']}"
            )

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as e:
            # Try to get page state even on error
            try:
                browser = await ensure_browser(session)
                state = await get_page_state(browser, session)
                error_context = f"\n\nCurrent state:\nScreenshot: {state['screenshot_path']}"
            except Exception:
                error_context = ""

            return {
                "content": [{"type": "text", "text": f"Click error: {str(e)}{error_context}"}],
                "is_error": True
            }

    @tool(
        "browser_type",
        "Type text into an input field. If role/name provided, finds and focuses that element first. "
        "Otherwise types into the currently focused element. Set press_enter=true to submit after typing.",
        {
            "text": str,
            "role": str,  # Optional
            "name": str,  # Optional
            "press_enter": bool,  # Optional, defaults to False
        }
    )
    async def browser_type(args: dict[str, Any]) -> dict[str, Any]:
        """Type text into an input field."""
        text = args.get("text", "")
        role = args.get("role", "")
        name = args.get("name", "")
        press_enter = args.get("press_enter", False)

        if not text and not press_enter:
            return {
                "content": [{"type": "text", "text": "Error: text is required (unless just pressing enter)"}],
                "is_error": True
            }

        try:
            browser = await ensure_browser(session)

            if browser.is_cdp and browser.cdp_page:
                # Use cdp-browser type
                if role:
                    try:
                        await browser.cdp_page.type(role, name=name if name else None, text=text, timeout=10.0)
                    except CDPElementNotFoundError:
                        state = await get_page_state(browser, session)
                        return {
                            "content": [{
                                "type": "text",
                                "text": f"No element found with role='{role}'" + (f" name='{name}'" if name else "") +
                                        f"\n\nScreenshot: {state['screenshot_path']}\n\n"
                                        f"Accessibility Tree:\n{state['accessibility_tree']}"
                            }],
                            "is_error": True
                        }
                if press_enter:
                    await browser.cdp_page.press("Enter")
            elif browser.pw_page:
                # Use Playwright
                if role:
                    if name:
                        locator = browser.pw_page.get_by_role(role, name=name)
                    else:
                        locator = browser.pw_page.get_by_role(role)

                    count = await locator.count()
                    if count == 0:
                        state = await get_page_state(browser, session)
                        return {
                            "content": [{
                                "type": "text",
                                "text": f"No element found with role='{role}'" + (f" name='{name}'" if name else "") +
                                        f"\n\nScreenshot: {state['screenshot_path']}\n\n"
                                        f"Accessibility Tree:\n{state['accessibility_tree']}"
                            }],
                            "is_error": True
                        }

                    if text:
                        await locator.first.fill(text)
                    if press_enter:
                        await locator.first.press("Enter")
                else:
                    if text:
                        await browser.pw_page.keyboard.type(text)
                    if press_enter:
                        await browser.pw_page.keyboard.press("Enter")

            # Wait for any updates
            await asyncio.sleep(0.5)

            state = await get_page_state(browser, session)

            action = f"Typed: \"{text[:50]}{'...' if len(text) > 50 else ''}\""
            if press_enter:
                action += " + Enter"
            if role:
                action += f" into [{role}]" + (f" \"{name}\"" if name else "")

            result_text = (
                f"{action}\n"
                f"URL: {state['current_url']}\n"
                f"Screenshot: {state['screenshot_path']}\n\n"
                f"Accessibility Tree:\n{state['accessibility_tree']}"
            )

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Type error: {str(e)}"}],
                "is_error": True
            }

    @tool(
        "browser_scroll",
        "Scroll the page. Direction can be 'up', 'down', 'top', or 'bottom'. "
        "Optionally specify pixels to scroll (default 500).",
        {
            "direction": str,
            "pixels": int,  # Optional, defaults to 500
        }
    )
    async def browser_scroll(args: dict[str, Any]) -> dict[str, Any]:
        """Scroll the page."""
        direction = args.get("direction", "down").lower()
        pixels = args.get("pixels", 500)

        valid_directions = ["up", "down", "top", "bottom"]
        if direction not in valid_directions:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: direction must be one of {valid_directions}"
                }],
                "is_error": True
            }

        try:
            browser = await ensure_browser(session)

            # Get the page object (either CDP or Playwright)
            if browser.is_cdp and browser.cdp_page:
                cdp_page = browser.cdp_page

                if direction == "top":
                    await cdp_page.evaluate("window.scrollTo(0, 0)")
                elif direction == "bottom":
                    await cdp_page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                elif direction == "up":
                    await cdp_page.evaluate(f"window.scrollBy(0, -{pixels})")
                else:  # down
                    await cdp_page.evaluate(f"window.scrollBy(0, {pixels})")

                # Wait for any lazy-loaded content
                await asyncio.sleep(0.3)

                state = await get_page_state(browser, session)

                # Get scroll position (use direct object literal, not arrow function for CDP)
                scroll_pos = await cdp_page.evaluate(
                    "({ x: window.scrollX, y: window.scrollY, height: document.body.scrollHeight })"
                )
            elif browser.pw_page:
                pw_page = browser.pw_page

                if direction == "top":
                    await pw_page.evaluate("window.scrollTo(0, 0)")
                elif direction == "bottom":
                    await pw_page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                elif direction == "up":
                    await pw_page.evaluate(f"window.scrollBy(0, -{pixels})")
                else:  # down
                    await pw_page.evaluate(f"window.scrollBy(0, {pixels})")

                # Wait for any lazy-loaded content
                await asyncio.sleep(0.3)

                state = await get_page_state(browser, session)

                # Get scroll position
                scroll_pos = await pw_page.evaluate(
                    "() => ({ x: window.scrollX, y: window.scrollY, height: document.body.scrollHeight })"
                )
            else:
                return {
                    "content": [{"type": "text", "text": "Error: no page available"}],
                    "is_error": True
                }

            result_text = (
                f"Scrolled {direction}" + (f" {pixels}px" if direction in ["up", "down"] else "") + "\n"
                f"Scroll position: y={scroll_pos['y']} / {scroll_pos['height']}\n"
                f"Screenshot: {state['screenshot_path']}\n\n"
                f"Accessibility Tree:\n{state['accessibility_tree']}"
            )

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Scroll error: {str(e)}"}],
                "is_error": True
            }

    @tool(
        "browser_close",
        "Close the browser session. Call when done with browser automation to free resources.",
        {}
    )
    async def browser_close(args: dict[str, Any]) -> dict[str, Any]:
        """Close the browser session."""
        try:
            if session.browser_session:
                await session.browser_session.close()
                session.browser_session = None
                return {"content": [{"type": "text", "text": "Browser session closed."}]}
            else:
                return {"content": [{"type": "text", "text": "No browser session was open."}]}
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error closing browser: {str(e)}"}],
                "is_error": True
            }

    return create_sdk_mcp_server(
        name="browser-tools",
        version="1.0.0",
        tools=[
            browser_navigate,
            browser_snapshot,
            browser_click,
            browser_type,
            browser_scroll,
            browser_close,
        ]
    )
