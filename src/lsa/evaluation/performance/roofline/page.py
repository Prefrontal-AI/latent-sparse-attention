"""Write the roofline page - one self-contained HTML file with the roofline JSON embedded.

python -m lsa.evaluation.performance.roofline.page DIR/roofline.json DIR/roofline.html
python -m lsa.evaluation.performance.roofline.page DIR/roofline.json DIR/roofline.html --compare H200/roofline.json

The page draws both arms of the profile, CuTeDSL first with the Triton arm beside it. With `--compare`, another GPU's
profile rides along, and the page sets each step's time on the two GPUs side by side.
"""

import argparse
import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

TEMPLATE = "page.html"
PLACEHOLDER = "{{ROOFLINE_JSON}}"
COMPARE_PLACEHOLDER = "{{COMPARE_JSON}}"


def drop_non_finite(value: Any) -> Any:
    """Replace NaN and infinities with None, which JSON can carry.

    Args:
        value: A parsed JSON value.

    Returns:
        The same structure with every non-finite float as None.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: drop_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [drop_non_finite(item) for item in value]
    return value


def embed(roofline: Any) -> str:
    """Serialize the roofline for a script tag.

    Args:
        roofline: The parsed roofline JSON.

    Returns:
        JSON text that cannot close the script tag or break a line.
    """
    text = json.dumps(drop_non_finite(roofline), separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def read_roofline(roofline: Path) -> Any:
    """Read a roofline JSON.

    Args:
        roofline: The `roofline.json` the report wrote.

    Raises:
        ValueError: The file is not valid JSON.
    """
    try:
        return json.loads(roofline.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{roofline} is not valid JSON: {error}") from error


def render(roofline: Path, output: Path, compare: Path | None = None) -> Path:
    """Render the page for a roofline JSON.

    Args:
        roofline: The `roofline.json` the report wrote.
        output: The HTML file to write.
        compare: Another GPU's `roofline.json`, to set each step's time beside this one's - None for none.

    Returns:
        The HTML file's path.

    Raises:
        ValueError: A roofline file is not valid JSON.
    """
    data = read_roofline(roofline)
    other = read_roofline(compare) if compare is not None else None
    template = resources.files(__package__).joinpath(TEMPLATE).read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    html = template.replace(PLACEHOLDER, embed(data)).replace(COMPARE_PLACEHOLDER, embed(other))
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    """Render the page from the command line."""
    parser = argparse.ArgumentParser(description="Render the roofline page.")
    parser.add_argument("roofline", type=Path, help="The roofline.json to embed.")
    parser.add_argument("output", type=Path, help="The HTML file to write.")
    parser.add_argument("--compare", type=Path, help="Another GPU's roofline.json to set beside this one.")
    arguments = parser.parse_args()
    print(render(arguments.roofline, arguments.output, arguments.compare))


if __name__ == "__main__":
    main()
