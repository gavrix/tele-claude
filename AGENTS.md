# AGENTS.md

## Project Overview
Telegram bot bridging forum topics to Claude Agent SDK sessions. Each topic = one Claude session.

## Architecture
- `bot.py` - Telegram bot setup, handlers registration
- `session.py` - Claude SDK integration, message streaming, tool permissions
- `handlers.py` - Telegram message/callback handlers
- `config.py` - Environment config (BOT_TOKEN, PROJECTS_DIR, browser settings)
- `logger.py` - Session logging
- `diff_image.py` - Syntax-highlighted edit diffs
- `browser_tools.py` - Playwright-based browser automation MCP tools

## Key Patterns
- Sessions stored in `sessions: dict[int, ClaudeSession]` (keyed by thread_id)
- Permission system: `DEFAULT_ALLOWED_TOOLS` + persistent `tool_allowlist.json`
- Streaming responses with rate limiting to avoid Telegram flood control
- HTML formatting via mistune for Telegram messages

## When making changes
- Run `pytest` before committing
- Run `pyright` for type checking
- Keep Telegram message limits in mind (4000 chars max)

## Dependencies
- python-telegram-bot (async)
- claude-agent-sdk
- mistune (markdown to HTML)
- Pillow (diff images)
- playwright (browser automation)

