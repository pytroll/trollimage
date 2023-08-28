"""Custom sphinx directive work with colormap objects.

This module must be on the ``sys.path`` and added to the list of extensions
in a ``conf.py``.

"""
import os
import importlib
from typing import Any

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from trollimage import colormap
from trollimage.image import Image


def setup(app):
    """Add custom extension to sphinx."""
    app.add_directive("trollimage_colormap", SingleColormap)

    return {
        "version": "0.1",
        "parallel_read": True,
        "parallel_write": True,
    }


class SingleColormap(SphinxDirective):
    """Custom sphinx directive for generating one or more colormap images."""

    has_content: bool = True
    option_spec: dict[str, Any] = {
        "colormap": str,
    }

    def run(self) -> list[nodes.Node]:
        """Import and generate colormap images to be inserted into the document."""
        cmap_name = self.options["colormap"]
        cmap_module_name, cmap_var = cmap_name.rsplit(".", 1)
        cmap_module = importlib.import_module(cmap_module_name)
        cmap_objects = getattr(cmap_module, cmap_var)
        if not isinstance(cmap_objects, (list, dict)):
            cmap_objects = {cmap_var: cmap_objects}
        if not isinstance(cmap_objects, dict):
            cmap_names = []
            for colormap_object in cmap_objects:
                cmap_name = [cmap_name for cmap_name, cmap_obj in cmap_module.__dict__.items()
                             if cmap_obj is colormap_object][0]
                cmap_names.append(cmap_name)
            cmap_objects = dict(zip(cmap_names, cmap_objects))

        image_nodes = []
        for cmap_name, cmap_obj in sorted(cmap_objects.items()):
            cb = colormap.colorbar(25, 500, cmap_obj)
            channels = [cb[i, :, :] for i in range(3)]
            im = Image(channels=channels, mode="RGB")
            cmap_fn = os.path.join("_static", "colormaps", f"{cmap_name}.png")
            im.save(cmap_fn)

            paragraph = nodes.paragraph(text=cmap_name)
            image = nodes.image(f".. image:: {cmap_fn}", **{"uri": cmap_fn, "alt": cmap_name})
            image_nodes += [paragraph, image]
        return image_nodes
