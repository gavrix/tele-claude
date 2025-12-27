class CDPError(Exception):
    """Base CDP error."""


class ConnectionError(CDPError):
    """Failed to connect to browser."""


class TimeoutError(CDPError):
    """Operation timed out."""


class NavigationError(CDPError):
    """Navigation failed (timeout, blocked, etc)."""


class ElementNotFoundError(CDPError):
    """Element matching criteria not found."""

    def __init__(self, role, name, tree_snippet: str, debug_file: str | None = None):
        super().__init__(f"Element with role={role!r}, name={name!r} not found")
        self.role = role
        self.name = name
        self.tree_snippet = tree_snippet
        self.debug_file = debug_file


class EvaluationError(CDPError):
    """JavaScript evaluation failed."""


class DialogError(CDPError):
    """Unexpected dialog (alert/confirm/prompt) blocked operation."""
