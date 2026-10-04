# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
"""Draw the xmmutablemap logo: a frozen dict.

A snowflake between a dict's braces: a mapping that cannot change, in
GalacticDynamics' purple and teal. The shapes are vector, so the logo is written
as an SVG, sharp at any size; for a bitmap, name a .png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import math
from pathlib import Path

TEAL, PURPLE = "#66a19a", "#7738eb"  # GalacticDynamics' colours

# In a 64-unit square. Each brace: its ends' x, its top and bottom, its depth.
BRACES = (15, 49, 10, 54, 3.4)
# The snowflake: its centre, its arms' half-length, and where along each arm
# its pair of side branches grows, and how long they are.
CENTRE, ARM, BRANCH_AT, BRANCH = 32, 13, 8, 4.5

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="512" height="512">
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="{braces}" stroke="{purple}" stroke-width="4.5"/>
    <path d="{arms}" stroke="{teal}" stroke-width="3.6"/>
    <path d="{branches}" stroke="{teal}" stroke-width="2.6"/>
  </g>
</svg>
"""


def brace(x: float, sign: int) -> str:
    """Return one brace as an SVG path: ``sign`` 1 opens right, -1 opens left."""
    _, _, top, bottom, depth = BRACES
    h, mid = bottom - top, (top + bottom) / 2
    # The brace's back, its point, and where its curves pull in to the point.
    back, tip, neck = x - sign * depth, x - sign * 2.2 * depth, x - sign * 1.8 * depth
    return (
        f"M{x:g} {top:g}"
        f"C{back:g} {top:g} {back:g} {top + 0.08 * h:g} {back:g} {top + 0.2 * h:g}"
        f"V{mid - 0.12 * h:g}"
        f"C{back:g} {mid - 0.03 * h:g} {neck:g} {mid:g} {tip:g} {mid:g}"
        f"C{neck:g} {mid:g} {back:g} {mid + 0.03 * h:g} {back:g} {mid + 0.12 * h:g}"
        f"V{bottom - 0.2 * h:g}"
        f"C{back:g} {bottom - 0.08 * h:g} {back:g} {bottom:g} {x:g} {bottom:g}"
    )


def snowflake() -> tuple[str, str]:
    """Return the snowflake's three arms, and its side branches, as SVG paths."""
    arms, branches = [], []
    for k in range(3):
        angle = math.radians(90 + 60 * k)
        dx, dy = ARM * math.cos(angle), ARM * math.sin(angle)
        arms.append(
            f"M{CENTRE - dx:.2f} {CENTRE - dy:.2f}L{CENTRE + dx:.2f} {CENTRE + dy:.2f}"
        )
        for end in (0, math.pi):  # both ends of the arm
            out = angle + end
            bx = CENTRE + BRANCH_AT * math.cos(out)
            by = CENTRE + BRANCH_AT * math.sin(out)
            for spread in (30, -30):
                a = out + math.radians(spread)
                branches.append(
                    f"M{bx:.2f} {by:.2f}"
                    f"L{bx + BRANCH * math.cos(a):.2f} {by + BRANCH * math.sin(a):.2f}"
                )
    return "".join(arms), "".join(branches)


def svg() -> str:
    """Return the logo as SVG text."""
    left, right, *_ = BRACES
    arms, branches = snowflake()
    return SVG.format(
        braces=brace(left, 1) + brace(right, -1),
        arms=arms,
        branches=branches,
        purple=PURPLE,
        teal=TEAL,
    )


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # noqa: PLC0415  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
