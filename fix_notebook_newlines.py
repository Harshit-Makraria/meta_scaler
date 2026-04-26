"""
Fix train_colab_3b_conversational.ipynb source format.

The notebook was stored with double-escaped newlines (literal \\n two chars)
inside each source list element, instead of proper Jupyter format where each
element is a separate line ending with an actual newline character.

This script:
1. For each code cell, joins all source elements into one string
2. Replaces literal \\n (two chars) with actual newline
3. Strips any trailing backslash left at end of the cell
4. Re-splits into proper per-line list format (each line ends with real \n
   except the last)
"""
import json
import pathlib

NB_PATH = pathlib.Path(
    r"c:\Users\harsh\Claude Cowork\meta_scaler\training\train_colab_3b_conversational.ipynb"
)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells_fixed = 0

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue

    src = cell["source"]
    if not src:
        continue

    # Join all elements
    joined = "".join(src)

    # Check if literal \n (two chars) are present — the bug
    if "\\n" in joined:
        # Replace literal \n with actual newline
        fixed = joined.replace("\\n", "\n")

        # Strip trailing backslash that was left at end of each cell
        # (the cell last-line had a trailing \ before the closing quote)
        fixed = fixed.rstrip("\\")

        # Re-split into Jupyter standard: each line as separate element,
        # all lines end with \n except the last
        lines = fixed.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            pass  # last line has no trailing \n — correct
        elif lines and lines[-1].endswith("\n"):
            # Remove trailing \n from last line for standard format
            lines[-1] = lines[-1].rstrip("\n")

        cell["source"] = lines
        cells_fixed += 1

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Fixed {cells_fixed} code cells.")
print("Done — notebook source format corrected.")
