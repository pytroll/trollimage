"""Helper script to convert colormaps from https://colorbrewer2.org into trollimage Colormap code.

The text output by this script should be copied to the ``trollimage/colormap.py``.

"""

import json
import sys
import urllib.request

JSON_URL = "https://raw.githubusercontent.com/axismaps/colorbrewer/master/export/colorbrewer.json"


def main():
    """Print python code version of trollimage Colormap objects for each colorbrewer colormap."""
    cmap_groups = _load_colormap_info_from_colorbrewer()
    _print_colormap_group(cmap_groups["seq"], "Sequential", "sequential_colormaps")
    _print_colormap_group(cmap_groups["div"], "Diverging", "diverging_colormaps")
    _print_colormap_group(cmap_groups["qual"], "Qualitative", "qualitative_colormaps",
                          normalize_values=False)


def _load_colormap_info_from_colorbrewer() -> dict[str, dict[str, list]]:
    with urllib.request.urlopen(JSON_URL) as json_file:  # nosec: B310
        colorbrewer_dict = json.load(json_file)
        cmap_groups: dict[str, dict[str, list]] = {"div": {}, "seq": {}, "qual": {}}
        for cmap_name, cmap_info in colorbrewer_dict.items():
            max_colors = max((num_colors_str for num_colors_str in cmap_info.keys() if num_colors_str != "type"),
                             key=lambda num_color_str: int(num_color_str))
            cmap_colors = cmap_info[max_colors]
            color_tuples = [rgb_color_str.replace("rgb(", "").replace(")", "").split(",")
                            for rgb_color_str in cmap_colors]
            cmap_groups[cmap_info["type"]][cmap_name.lower()] = color_tuples
    return cmap_groups


def _print_colormap_group(cmap_group: dict[str, list], human_group_name: str, group_var_name: str,
                          normalize_values: bool = True) -> None:
    print(f"# * {human_group_name} Colormaps *\n")
    for cmap_name, cmap_colors in sorted(cmap_group.items()):
        cmap_values, color_human_strings = _color_info_as_human_friendly_strings(cmap_colors, normalize_values)
        _print_single_colormap(cmap_name, cmap_values, color_human_strings)

    print(f"{group_var_name} = {{")
    for cmap_name in sorted(cmap_group.keys()):
        print(f"    \"{cmap_name}\": {cmap_name},")
    print("}\n")


def _color_info_as_human_friendly_strings(
        cmap_colors: list[tuple[int, int, int]],
        normalize_values: bool,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    num_colors = len(cmap_colors)
    color_human_strings = [
        (f"{rgb_color[0]} / 255", f"{rgb_color[1]} / 255", f"{rgb_color[2]} / 255")
        for rgb_color in cmap_colors
    ]
    cmap_values = [str(color_idx) for color_idx in range(num_colors)]
    if normalize_values:
        # 0 - 1 normalized values for non-qualitative colormaps
        cmap_values = [f"{cval} / {num_colors - 1}" for cval in cmap_values]
    return cmap_values, color_human_strings


def _print_single_colormap(cmap_name: str, cmap_values: list[str], cmap_colors: list[tuple[str, str, str]]) -> None:
    cmap_pairs = [(cval, rgb_color) for cval, rgb_color in zip(cmap_values, cmap_colors)]
    print(f"{cmap_name} = Colormap(")
    for cmap_value_str, cmap_color_tuple in cmap_pairs:
        print(f"    ({cmap_value_str}, "
              f"({cmap_color_tuple[0]}, {cmap_color_tuple[1]}, {cmap_color_tuple[2]})),")
    print(")\n")


if __name__ == "__main__":
    sys.exit(main())
