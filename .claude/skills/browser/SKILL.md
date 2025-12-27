---
name: browser
description: Browse websites, click buttons, fill forms, check notifications, take screenshots. Use when user asks to interact with any web page.
---

# Browser Automation

Use the browser MCP tools to interact with web pages.

## Tools

- `browser_navigate(url)` - Navigate to URL, returns accessibility tree + screenshot path
- `browser_click(role, name, index?)` - Click element by accessibility role/name
- `browser_type(text, role?, name?, press_enter?)` - Type into input fields
- `browser_scroll(direction, pixels?)` - Scroll page (up/down/top/bottom)
- `browser_snapshot()` - Get current page state without action
- `browser_close()` - Close browser session

## How to Target Elements

Elements are identified by their accessibility role and name from the tree:
- `link "Log in"` -> role="link", name="Log in"
- `button "Submit"` -> role="button", name="Submit"
- `textbox "Email"` -> role="textbox", name="Email"
- `combobox "Search"` -> role="combobox", name="Search"

If multiple elements match, use the `index` parameter (0-indexed).

## Workflow

1. Use `browser_navigate` to go to a URL
2. Read the returned accessibility tree to find elements
3. Use `browser_click` or `browser_type` to interact
4. Check the new accessibility tree after each action
5. Use Read tool on screenshot path to see the page visually

## Configuration

- If `BROWSER_CDP_ENDPOINT` is set, connects to user's running Chrome (with existing logins)
- Otherwise launches own browser with persistent context
- Each session reuses the same browser page across commands
