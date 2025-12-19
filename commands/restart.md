<!-- Restart the bot -->
Find the tmux pane running `python bot.py` and restart it.

Steps:
1. List tmux sessions and panes to find where bot.py is running
2. Send Ctrl+C followed by the start command in a single atomic tmux send-keys command

Use this pattern to restart atomically (Ctrl+C and new command in one send-keys):
```bash
tmux send-keys -t <session>:<window>.<pane> C-c "python bot.py" Enter
```

This ensures the bot restarts even though stopping it terminates the current session.
