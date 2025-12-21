"""
Browser automation tools for Claude Agent SDK.

Provides Playwright-based browser control with:
- Persistent context for cookies/authentication
- Accessibility tree extraction for element targeting
- Screenshot capture for visual understanding
"""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from claude_agent_sdk import create_sdk_mcp_server, tool

if TYPE_CHECKING:
    from session import ClaudeSession

# Configuration (imported from config.py at runtime to avoid circular imports)
ACCESSIBILITY_MAX_LENGTH = 50000  # Truncate if larger

# Global playwright instance (reused across sessions)
_playwright: Optional[Playwright] = None
_playwright_lock = asyncio.Lock()


async def get_playwright() -> Playwright:
    """Get or create global playwright instance."""
    global _playwright
    async with _playwright_lock:
        if _playwright is None:
            _playwright = await async_playwright().start()
        return _playwright


@dataclass
class BrowserSession:
    """Manages browser state for a Claude session."""
    context: BrowserContext
    page: Page
    user_data_dir: Optional[Path] = None
    browser: Optional[Browser] = None  # Only set for CDP connections
    last_activity: float = field(default_factory=time.time)
    is_cdp: bool = False  # True if connected via CDP

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    async def close(self) -> None:
        """Close browser context (but not CDP browser - that's the user's Chrome)."""
        try:
            if self.is_cdp:
                # For CDP, just close the page we created, not the whole browser
                await self.page.close()
            else:
                await self.context.close()
        except Exception:
            pass


async def create_browser_session(thread_id: int, headless: bool = True, data_dir: Optional[Path] = None) -> BrowserSession:
    """Create new browser session with persistent context or CDP connection.

    Args:
        thread_id: Telegram thread ID for isolation
        headless: Run browser in headless mode (ignored for CDP)
        data_dir: Base directory for browser data persistence (ignored for CDP)
    """
    from config import BROWSER_CDP_ENDPOINT, BROWSER_DATA_DIR, BROWSER_HEADLESS

    pw = await get_playwright()

    # If CDP endpoint is configured, connect to existing Chrome
    if BROWSER_CDP_ENDPOINT:
        browser = await pw.chromium.connect_over_cdp(BROWSER_CDP_ENDPOINT)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()

        return BrowserSession(
            context=context,
            page=page,
            browser=browser,
            is_cdp=True
        )

    # Otherwise launch our own browser with persistent context
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
        # Enable accessibility tree
        args=["--force-renderer-accessibility"]
    )

    # Get the default page or create one
    page = context.pages[0] if context.pages else await context.new_page()

    return BrowserSession(
        context=context,
        page=page,
        user_data_dir=user_data_dir
    )


async def ensure_browser(session: "ClaudeSession") -> BrowserSession:
    """Ensure browser session exists, creating if needed."""
    from config import BROWSER_HEADLESS
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


async def get_accessibility_tree(page: Page) -> str:
    """Extract accessibility tree from page using ARIA snapshot.

    Uses the modern aria_snapshot() method which returns a YAML-like
    representation of the accessibility tree.
    """
    try:
        # Use aria_snapshot on body - returns YAML-like accessibility tree
        aria_snapshot = await page.locator("body").aria_snapshot()
        if aria_snapshot:
            # Truncate if too large
            if len(aria_snapshot) > ACCESSIBILITY_MAX_LENGTH:
                aria_snapshot = aria_snapshot[:ACCESSIBILITY_MAX_LENGTH] + "\n... (truncated)"
            return aria_snapshot
        return "(empty page)"
    except Exception as e:
        return f"(accessibility tree unavailable: {e})"


async def take_screenshot(page: Page, session: "ClaudeSession") -> str:
    """Take screenshot and return temp file path."""
    import tempfile
    screenshot_path = Path(tempfile.gettempdir()) / f"browser_screenshot_{session.thread_id}_{int(time.time() * 1000)}.png"
    await page.screenshot(path=str(screenshot_path))
    return str(screenshot_path)


async def get_page_state(page: Page, session: "ClaudeSession") -> dict[str, str]:
    """Get current page state including accessibility tree and screenshot."""
    tree = await get_accessibility_tree(page)
    screenshot_path = await take_screenshot(page, session)

    return {
        "accessibility_tree": tree,
        "screenshot_path": screenshot_path,
        "current_url": page.url,
        "page_title": await page.title(),
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
            await browser.page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Brief wait for dynamic content (don't use networkidle - sites like Twitter never idle)
            await asyncio.sleep(1)

            state = await get_page_state(browser.page, session)

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
            state = await get_page_state(browser.page, session)

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

            # Find element by role and optional name
            if name:
                locator = browser.page.get_by_role(role, name=name)
            else:
                locator = browser.page.get_by_role(role)

            count = await locator.count()
            if count == 0:
                # Get current state to help user find the right element
                state = await get_page_state(browser.page, session)
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

            # Click the element
            await locator.nth(index).click()

            # Wait for any navigation or dynamic updates
            await browser.page.wait_for_load_state("domcontentloaded", timeout=5000)

            state = await get_page_state(browser.page, session)

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
                state = await get_page_state(browser.page, session)
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

            # If role provided, find and interact with that element
            if role:
                if name:
                    locator = browser.page.get_by_role(role, name=name)
                else:
                    locator = browser.page.get_by_role(role)

                count = await locator.count()
                if count == 0:
                    state = await get_page_state(browser.page, session)
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"No element found with role='{role}'" + (f" name='{name}'" if name else "") +
                                    f"\n\nScreenshot: {state['screenshot_path']}\n\n"
                                    f"Accessibility Tree:\n{state['accessibility_tree']}"
                        }],
                        "is_error": True
                    }

                # Fill the element (clears existing text first)
                if text:
                    await locator.first.fill(text)

                if press_enter:
                    await locator.first.press("Enter")
            else:
                # Type into currently focused element
                if text:
                    await browser.page.keyboard.type(text)
                if press_enter:
                    await browser.page.keyboard.press("Enter")

            # Wait for any updates
            await asyncio.sleep(0.5)

            state = await get_page_state(browser.page, session)

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

            if direction == "top":
                await browser.page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction == "up":
                await browser.page.evaluate(f"window.scrollBy(0, -{pixels})")
            else:  # down
                await browser.page.evaluate(f"window.scrollBy(0, {pixels})")

            # Wait for any lazy-loaded content
            await asyncio.sleep(0.3)

            state = await get_page_state(browser.page, session)

            # Get scroll position
            scroll_pos = await browser.page.evaluate(
                "() => ({ x: window.scrollX, y: window.scrollY, height: document.body.scrollHeight })"
            )

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
