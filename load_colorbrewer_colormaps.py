"""Helper script to convert colormaps from https://colorbrewer2.org into trollimage Colormap code.

The text output by this script should be copied to the ``trollimage/colormap.py``.

"""

import json
import sys
import urllib.request

JSON_URL = "https://raw.githubusercontent.com/axismaps/colorbrewer/master/export/colorbrewer.json"


def main():
    """Print python code version of trollimage Colormap objects for each colorbrewer colormap."""
    with urllib.request.urlopen(JSON_URL) as json_file:
        colorbrewer_dict = json.load(json_file)
        cmap_groups = {"div": {}, "seq": {}, "qual": {}}
        for cmap_name, cmap_info in colorbrewer_dict.items():
            max_colors = max(num_colors_str for num_colors_str in cmap_info.keys() if num_colors_str != "type")
            cmap_groups[cmap_info["type"]][cmap_name.lower()] = cmap_info[max_colors]
        for group_name in ("seq", "div", "qual"):
            human_group_name = {"seq": "Sequential", "div": "Diverging", "qual": "Qualitative"}[group_name]
            print(f"# * {human_group_name} Colormaps *\n")
            cmap_group = cmap_groups[group_name]
            for cmap_name, cmap_colors in sorted(cmap_group.items()):
                num_colors = len(cmap_colors)
                colors = [rgb_color_str.replace("rgb(", "").replace(")", "").split(",")
                          for rgb_color_str in cmap_colors]
                norm_colors = [tuple(f"{color} / 255" for color in rgb_color) for rgb_color in colors]
                cmap_values = [str(color_idx) for color_idx in range(num_colors)]
                if group_name != "qual":
                    # 0 - 1 normalized values for non-qualitative colormaps
                    cmap_values = [f"{cval} / {num_colors}" for cval in cmap_values]
                cmap_pairs = [(cval, rgb_color) for cval, rgb_color in zip(cmap_values, norm_colors)]
                print(f"{cmap_name} = Colormap(")
                for cmap_value_str, cmap_color_tuple in cmap_pairs:
                    print(f"    ({cmap_value_str},"
                          f"({cmap_color_tuple[0]}, {cmap_color_tuple[1]}, {cmap_color_tuple[2]})),")
                print(")\n")

            group_var_name = {
                "seq": "sequential_colormaps",
                "div": "diverging_colormaps",
                "qual": "qualitative_colormaps",
            }[group_name]
            print(f"{group_var_name} = {{")
            for cmap_name in sorted(cmap_group.keys()):
                print(f"    \"{cmap_name}\": {cmap_name},")
            print("}\n")


if __name__ == "__main__":
    sys.exit(main())
