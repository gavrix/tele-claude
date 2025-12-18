# Claude Code TUI Parser Design

## Overview

This document describes the color-based segmentation approach for parsing Claude Code CLI output.

## Key Insight

Claude Code uses **consistent color coding** for different content types. We can use pyte's `screen.buffer` which preserves color attributes per character to reliably classify content.

## Color Mapping

| Color (hex) | RGB | Meaning | Action |
|-------------|-----|---------|--------|
| `d77757` | 215,119,87 | **Orange** - Thinking/Generating spinner | `thinking` segment (transient) |
| `eb9f7f` | 235,159,127 | **Light orange** - Part of thinking animation | `thinking` segment (transient) |
| `999999` | 153,153,153 | **Grey** - Metadata, hints, tool calls (in progress) | `tool_call` or skip (UI noise) |
| `4eba65` | 78,186,101 | **Green** - Completed tool calls | `tool_call` segment |
| `ffffff` | 255,255,255 | **White** - Bullets ⏺, banner stars, user echo text | Context-dependent |
| `b1b9f9` | 177,185,249 | **Light blue/purple** - Box drawing lines | Skip (decorative) |
| `add8e6` | 173,216,230 | **Light blue** - Banner decoration | `banner` segment |
| `default` | - | **Default terminal color** - Claude's response text | `response` segment |

### Background Colors

| BG Color (hex) | RGB | Meaning |
|----------------|-----|---------|
| `373737` | 55,55,55 | **Dark grey** - User echo line background |

## Important Design Principle: No Silent Skipping

**NEVER silently skip unclassified content.** If a line doesn't match any known classification rule, put it into an `unclassified` segment instead of dropping it. This allows:

1. Visibility into what the parser is missing
2. Opportunity to update classifier with new patterns
3. No lost content during development

```python
# WRONG - silently drops unknown content
if not matches_any_rule(line):
    return None, None  # ❌ Lost forever

# RIGHT - preserves unknown content for debugging
if not matches_any_rule(line):
    return 'unclassified', content  # ✅ Visible, can improve classifier
```

Only skip content that is **explicitly identified** as noise (user echo, box drawing, etc.) - never skip "unknown" content.

## Segmentation Rules

### 1. User Echo Detection
```
IF line has background color 373737 (dark grey)
AND line starts with ">"
THEN → skip (user_echo)
```

### 2. Thinking/Generating Detection
```
IF first non-space char has foreground d77757 or eb9f7f (orange)
THEN → thinking segment (transient=True)
```

### 3. Status/Metadata Detection
```
IF line foreground is 999999 (grey)
AND contains "Thought for" OR "shortcuts" OR version info
THEN → skip or status segment
```

### 4. Response Detection
```
IF line starts with white (ffffff) bullet ⏺
AND rest of line is default color
THEN → response segment (transient=False)
```

### 5. Box Drawing Detection
```
IF line foreground is b1b9f9 (light blue)
OR line contains mostly box chars (─│┌┐└┘├┤┬┴┼╭╮╰╯)
THEN → skip (decorative)
```

### 6. Tool Call Detection
```
# New format (current Claude Code):
IF line starts with ⏺ bullet (grey or green)
AND followed by ToolName( or ToolName space
THEN → tool_call segment with "🔧 ToolName(args)"

# Old format (box-based):
IF line contains box chars AND tool name (Read, Write, Edit, Bash, etc.)
THEN → tool_call segment
```

### 7. UI Noise Filtering
```
Skip lines containing:
- "Tip:" messages
- "Thought for Xs (ctrl+o to show thinking)"
- "ctrl+o to expand"
- Lines starting with ⎿ or └ that contain "Running" or "Tip:"
- Empty prompts (> with no content)
- Shortcut hints (? for shortcuts)
- IDE hints (/ide for Cursor)
```

## Pyte Buffer Access

Pyte preserves color info in `screen.buffer`:

```python
screen = pyte.Screen(cols, rows)
stream = pyte.Stream(screen)
stream.feed(raw_output)

# Access character with color
for y in range(rows):
    line = screen.buffer[y]
    for x, char in line.items():
        text = char.data      # The character
        fg = char.fg          # Foreground color (hex string or 'default')
        bg = char.bg          # Background color (hex string or 'default')
```

## Implementation

### Color Constants

```python
# Foreground colors
COLOR_ORANGE = 'd77757'        # Thinking/generating spinner
COLOR_LIGHT_ORANGE = 'eb9f7f'  # Thinking animation
COLOR_GREY = '999999'          # Metadata, hints, in-progress tools
COLOR_GREEN = '4eba65'         # Completed tool calls
COLOR_WHITE = 'ffffff'         # Bullets, banner stars
COLOR_LIGHT_BLUE = 'b1b9f9'    # Box drawing lines
COLOR_BANNER_BLUE = 'add8e6'   # Banner decoration
COLOR_DEFAULT = 'default'      # Terminal default - response text

# Background colors
BG_USER_ECHO = '373737'        # User echo line (dark grey)
```

### Line Classification Function

```python
def classify_line(screen, line_num: int) -> tuple[str, str]:
    """
    Classify a line based on its color.
    Returns (segment_type, content) or (None, None) to skip.
    """
    line_buffer = screen.buffer[line_num]
    if not line_buffer:
        return None, None

    # Build line content and get first char's color
    chars = []
    first_fg = None
    first_bg = None

    for x in sorted(line_buffer.keys()):
        char = line_buffer[x]
        if char.data.strip() and first_fg is None:
            first_fg = char.fg
            first_bg = char.bg
        chars.append(char.data)

    content = ''.join(chars).strip()
    if not content:
        return None, None

    # User echo: dark grey background + starts with >
    if first_bg == BG_USER_ECHO and content.startswith('>'):
        return 'user_echo', content

    # Thinking: orange foreground
    if first_fg in (COLOR_ORANGE, COLOR_LIGHT_ORANGE):
        return 'thinking', content

    # Grey metadata/hints
    if first_fg == COLOR_GREY:
        return 'status', content

    # Box drawing (light blue)
    if first_fg == COLOR_LIGHT_BLUE:
        return 'box', content

    # Response: white bullet followed by default text
    if content.startswith('⏺'):
        return 'response', content

    # Tool detection (has box chars + tool name)
    if has_tool_header(content):
        return 'tool_call', content

    # Response: default terminal color (no special coloring)
    if first_fg == COLOR_DEFAULT:
        return 'response', content

    # IMPORTANT: Don't silently skip - mark as unclassified for debugging
    return 'unclassified', content
```

### Updated Parser Structure

```python
class OutputParser:
    def __init__(self, cols=120, rows=500):
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.prev_lines = []
        self.segments = []

    def feed(self, raw_text: str) -> List[SegmentUpdate]:
        self.stream.feed(raw_text)

        # Get lines with color classification
        new_lines = self._get_classified_lines()

        # Find divergence
        divergence = self._find_divergence(self.prev_lines, new_lines)
        if divergence is None:
            return []

        # Parse from divergence using color info
        # ... rest of delta logic

    def _get_classified_lines(self) -> List[tuple]:
        """Get lines as (type, content, line_num) tuples."""
        result = []
        for y in range(len(self.screen.buffer)):
            seg_type, content = classify_line(self.screen, y)
            if seg_type and content:
                result.append((seg_type, content, y))
        return result
```

## Multiple Simultaneous Segments

Claude Code can update several areas at once:

```
┌────────────────────────────────────────┐
│ [stable] Banner                        │
│ [stable] Previous response             │
│ [updating] Current tool output         │ ← being filled
│ [updating] ✶ Generating...             │ ← spinner animating
│ [updating] ? for shortcuts             │ ← hint appearing/disappearing
└────────────────────────────────────────┘
```

The delta-based approach handles this by:
1. Finding earliest divergence point
2. Re-parsing everything from that point
3. Generating minimal updates for changed segments

## Files to Modify

- `/Users/gavrix/Projects/tele-claude/parser.py` - Add color-based classification
- `/Users/gavrix/Projects/tele-claude/session.py` - Handle SegmentUpdate actions
- `/Users/gavrix/Projects/tele-claude/test_parser.py` - Add color-based tests

## Segment Types

```python
SegmentType = Literal[
    "thinking",      # Orange spinner/status (transient)
    "response",      # Claude's actual response text
    "tool_call",     # Tool header (Read, Write, etc.)
    "tool_output",   # Content inside tool box
    "status",        # Grey metadata (version, "Thought for", etc.)
    "banner",        # Startup banner
    "unclassified",  # IMPORTANT: Unknown content for debugging
]

# These are identified and skipped (not segments):
# - user_echo: Dark grey bg + starts with >
# - box: Light blue box drawing lines
```

## Test Cases

1. **Thinking spinner** - Orange color, cycles through ✶✻✽✢·∴
2. **User echo** - Grey background, starts with > (skip)
3. **Response** - White bullet ⏺, default text color
4. **Tool call (new format)** - Grey/green ⏺ bullet + ToolName(args) → "🔧 Bash(git status)"
5. **Tool call (old format)** - Box chars + tool name
6. **Metadata** - Grey text (version, hints)
7. **Box lines** - Light blue, decorative (skip)
8. **UI Noise** - Tip messages, "Thought for", Running..., ctrl+o hints (skip)
9. **Unclassified** - Anything else → visible for debugging

## Example Final Output

For a typical interaction asking "What's the git status?":

```
[0] banner     - "* ▐▛███▜▌ * Claude Code v2.0.69..."
[1] response   - "⏺ Hey! Not much, just here..."
[2] tool_call  - "🔧 Bash(git status)"
[3] response   - "⏺ Here's your current git status:..."
```

Filtered out (not shown):
- "∴ Thought for 4s (ctrl+o to show thinking)"
- "⎿  Tip: Use /permissions..."
- "⎿  Running..."
- Thinking spinners (shown as typing indicator instead)
