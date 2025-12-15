import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
PROJECTS_DIR = Path(os.getenv("PROJECTS_DIR", Path.home() / "Projects"))
GENERAL_TOPIC_ID = 0
