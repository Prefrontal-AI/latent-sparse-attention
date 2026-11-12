"""Repo hygiene, not behaviour - relative links resolve and only the plain hyphen is used.

A link is `[text](path)` in a markdown file, a YAML comment, or a Python comment or docstring.

Run by `make lint` and by the commit hook. These are rules about the repository rather
than tests of the package, so they do not live in `tests/`, which covers what the code does.

    python tools/check_repo.py
"""

import io
import re
import subprocess
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)
INLINE_CODE = re.compile(r"`[^`\n]*`")
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
# The em and en dash, named by code point so this file passes its own check
DASH = re.compile(f"[{chr(0x2014)}{chr(0x2013)}]")

# A model's own output, quoted verbatim as a fixture - editing it would test something the model never wrote
DASH_ALLOWED = {"tests/unit/evaluation/accuracy/test_letters.py"}

# A benchmark's own text and the titles of real papers, hashed by the card that pins them - editing a dash would
# change the benchmark, so the kit is kept as it was written
DASH_ALLOWED_PREFIXES = ("data/papertrail/kit/",)


def git(*arguments: str) -> str:
    """Run git in the repository and return its stdout.

    Args:
        arguments: The git subcommand and its arguments.

    Raises:
        subprocess.CalledProcessError: git failed.
    """
    return subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=True, check=True).stdout


def tracked_files() -> list[str]:
    """Return every tracked path, relative to the repository."""
    return [name for name in git("ls-files", "-z").split("\0") if name]


def markdown_files() -> list[Path]:
    """Return every tracked markdown file."""
    return [REPO / name for name in tracked_files() if name.endswith(".md")]


def slug(heading: str) -> str:
    """Return the anchor GitHub gives a heading.

    Args:
        heading: The heading's text, without the leading hashes.
    """
    text = re.sub(r"`|\*|_", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def anchors(path: Path) -> set[str]:
    """Return the anchors a markdown file's headings define, ignoring fenced code.

    Args:
        path: The markdown file.
    """
    body = INLINE_CODE.sub("", FENCE.sub("", path.read_text()))
    return {slug(heading) for heading in HEADING.findall(body)}


def comment_text(source: Path) -> str:
    """Return the text of a YAML file's comments or of a Python file's comments and docstrings.

    Args:
        source: A `.yaml`, `.yml` or `.py` file.
    """
    body = source.read_text()
    if source.suffix != ".py":
        return "\n".join(line.partition("#")[2] for line in body.split("\n") if "#" in line)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(body).readline))
    except (tokenize.TokenError, IndentationError):
        return ""
    # A triple-quoted string is a docstring or prose, a one-line string may be code that only looks like a link
    prose = [
        t.string
        for t in tokens
        if t.type == tokenize.COMMENT or (t.type == tokenize.STRING and t.string[-3:] in ('"""', "'''"))
    ]
    return "\n".join(prose)


def link_sources() -> list[tuple[Path, str]]:
    """Return every tracked file that carries links, with the text its links are read from."""
    sources = []
    for name in tracked_files():
        path = REPO / name
        if name.endswith(".md"):
            sources.append((path, INLINE_CODE.sub("", FENCE.sub("", path.read_text()))))
        elif (
            name.endswith((".yaml", ".yml", ".py"))
            and name not in DASH_ALLOWED
            and not name.startswith(DASH_ALLOWED_PREFIXES)
        ):
            sources.append((path, INLINE_CODE.sub("", FENCE.sub("", comment_text(path)))))
    return sources


def broken_links() -> list[str]:
    """Return every relative link whose target or anchor does not resolve."""
    broken = []
    for source, body in link_sources():
        for target in LINK.findall(body):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            if not path_part:
                if anchor and source.suffix == ".md" and anchor not in anchors(source):
                    broken.append(f"{source.relative_to(REPO)} -> #{anchor} (no such heading here)")
                continue
            resolved = (source.parent / path_part).resolve()
            if not resolved.exists():
                broken.append(f"{source.relative_to(REPO)} -> {target} (no such path)")
            elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                broken.append(f"{source.relative_to(REPO)} -> {target} (no such heading)")
    return broken


def stray_dashes() -> list[str]:
    """Return every tracked line carrying an em or en dash, which AGENTS.md forbids."""
    offenders = []
    for name in tracked_files():
        if name in DASH_ALLOWED or name.startswith(DASH_ALLOWED_PREFIXES):
            continue
        try:
            body = (REPO / name).read_text()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for number, line in enumerate(body.split("\n"), start=1):
            if DASH.search(line):
                offenders.append(f"{name}:{number}  {line.strip()[:70]}")
    return offenders


def main() -> int:
    """Report every check and return the process exit code."""
    failed = 0
    checks = (
        ("broken links", broken_links()),
        ("em or en dashes", stray_dashes()),
    )
    for label, rows in checks:
        if rows:
            failed = 1
            print(f"{label}:")
            for row in rows:
                print(f"  {row}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
