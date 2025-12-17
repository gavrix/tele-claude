# tele-bot

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

## Setup

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Enable "Topics" in your Telegram group (group settings → Topics)
3. Add the bot to your group as admin

```bash
# Clone and install
git clone https://github.com/gavrix/tele-bot.git
cd tele-bot
pip install -r requirements.txt

# Configure
echo "BOT_TOKEN=your_bot_token_here" > .env

# Optionally set projects directory (defaults to ~/Projects)
echo "PROJECTS_DIR=/path/to/projects" >> .env

# Run
python bot.py
```

## Usage

1. Create a new topic in your Telegram forum group
2. Bot auto-detects and shows a folder picker
3. Select a project folder to bind to this topic
4. Chat with Claude in that topic

Or use `/new` command to manually start a session.

## Requirements

- Python 3.10+
- Telegram Bot API token
- Claude Code CLI installed and authenticated
