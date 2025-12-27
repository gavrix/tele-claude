# tele-claude

A Telegram bot that connects forum topics to Claude Code CLI sessions.

## How it works

- Each Telegram forum topic maps to one Claude session
- Messages are forwarded to Claude via the SDK
- Claude responses stream back as Telegram messages
- Tool calls displayed with 🔧 indicator
- Typing indicator shown during processing
- Edit diffs rendered as syntax-highlighted images
- Photo uploads supported for image analysis
- Interactive tool permission prompts (approve/deny)
- Context window warning when below 15%
- Send a new message to interrupt Claude mid-response
- Browser automation via CDP or Playwright

## Setup

### 1. Create a Telegram Bot

1. Open [@BotFather](https://t.me/botfather) in Telegram
2. Send `/newbot` and follow the prompts to name your bot
3. Copy the API token (looks like `123456789:ABCdefGHI...`)

### 2. Create a Forum Group

1. Create a new Telegram group (or use existing)
2. Go to group settings → Topics → Enable
3. Add your bot to the group
4. Promote bot to admin with these permissions:
   - Delete messages
   - Manage topics

### 3. Install and Run

```bash
# Clone and install
git clone https://github.com/gavrix/tele-claude.git
cd tele-claude
pip install -r requirements.txt

# Configure
echo "BOT_TOKEN=your_bot_token_here" > .env

# Optionally set projects directory (defaults to ~/Projects)
echo "PROJECTS_DIR=/path/to/projects" >> .env

# Run
python bot.py
```

## Usage

### Multi-project mode (default)

1. Create a new topic in your Telegram forum group
2. Bot auto-detects and shows a folder picker
3. Select a project folder to bind to this topic
4. Chat with Claude in that topic

Or use `/new` command to manually start a session.

### Local project mode

Run a bot instance anchored to a specific project directory:

```bash
cd /path/to/your/project

# Create config file
cat > .env.telebot << EOF
BOT_TOKEN=your_bot_token_here
ALLOWED_CHATS=-100xxxxx  # optional
EOF

# Run
python /path/to/tele-claude/bot_local.py
```

Every new topic auto-starts a session in that directory. Useful for running separate bot instances per project.

## Browser Automation

Claude can control a browser to navigate websites, click elements, fill forms, and take screenshots.

### Option 1: Use existing Chrome (recommended)

Connect to your running Chrome with all your cookies and logged-in sessions:

```bash
# Start Chrome with remote debugging
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Add to .env
echo "BROWSER_CDP_ENDPOINT=http://localhost:9222" >> .env
```

### Option 2: Standalone Chromium

If no CDP endpoint is configured, the bot launches its own Chromium instance with persistent storage per session.

## Requirements

- Python 3.10+
- Telegram Bot API token
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and logged in

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BOT_TOKEN` | Yes | - | Telegram bot token from BotFather |
| `PROJECTS_DIR` | No | `~/Projects` | Root directory for project folders |
| `ALLOWED_CHATS` | No | - | Comma-separated chat IDs to allow (empty = allow all) |
| `BROWSER_CDP_ENDPOINT` | No | - | Chrome DevTools Protocol endpoint (e.g., `http://localhost:9222`) |
| `BROWSER_HEADLESS` | No | `true` | Run standalone Chromium in headless mode |
| `BROWSER_DATA_DIR` | No | `~/.tele-bot/browsers` | Persistent storage for standalone browser sessions |
