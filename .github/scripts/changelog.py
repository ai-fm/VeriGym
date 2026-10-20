"""Maintain CHANGELOG.md for the release workflows.

python .github/scripts/changelog.py release 0.2.0   # move the [Unreleased] entries under 0.2.0
python .github/scripts/changelog.py notes 0.2.0     # print the 0.2.0 entries (release notes)
"""

import datetime
import sys
from pathlib import Path

CHANGELOG = Path(__file__).parents[2] / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"
# Empty groups that [Unreleased] starts with after every release
GROUPS = ["### Added", "### Changed", "### Fixed", "### Removed"]


def find_heading(lines: list[str], heading: str) -> int:
    """Return the index of the line starting with `heading`, e.g. "## [0.2.0]"."""
    for i, line in enumerate(lines):
        if line.startswith(heading):
            return i
    sys.exit(f"CHANGELOG.md has no '{heading}' heading")


def section_end(lines: list[str], start: int) -> int:
    """Return the index of the next version heading after `start` (or the end of the file)."""
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            return i
    return len(lines)


def section(lines: list[str], heading: str) -> str:
    """Return the entries below `heading`, up to the next version heading."""
    start = find_heading(lines, heading)
    return "\n".join(lines[start + 1 : section_end(lines, start)]).strip()


def drop_empty_groups(body: list[str]) -> list[str]:
    """Return `body` without the "### ..." groups that have no entries, tidied up."""
    blocks = [[]]  # lines before the first group, then one block per group
    for line in body:
        if line.startswith("### "):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    result = []
    for block in blocks:
        has_heading = bool(block) and block[0].startswith("### ")
        entries = "\n".join(block[1:] if has_heading else block).strip()
        if entries:
            result += ([block[0], ""] if has_heading else []) + entries.split("\n") + [""]
    return result


def release(version: str) -> None:
    lines = CHANGELOG.read_text().splitlines()
    if any(line.startswith(f"## [{version}]") for line in lines):
        sys.exit(f"CHANGELOG.md already has a section for {version}")
    start = find_heading(lines, UNRELEASED)
    end = section_end(lines, start)
    entries = drop_empty_groups(lines[start + 1 : end])
    if not entries:
        sys.exit(f"CHANGELOG.md has no entries under '{UNRELEASED}', add them first")
    today = datetime.date.today().isoformat()
    empty_groups = [line for group in GROUPS for line in (group, "")]
    lines[start:end] = [UNRELEASED, ""] + empty_groups + [f"## [{version}] - {today}", ""] + entries
    CHANGELOG.write_text("\n".join(lines).rstrip("\n") + "\n")


def notes(version: str) -> None:
    print(section(CHANGELOG.read_text().splitlines(), f"## [{version}]"))


if __name__ == "__main__":
    commands = {"release": release, "notes": notes}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2])
