from config import PROJECTS_DIR


def get_project_folders() -> list[str]:
    """List directories in ~/Projects."""
    if not PROJECTS_DIR.exists():
        return []
    return sorted([
        d.name for d in PROJECTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
