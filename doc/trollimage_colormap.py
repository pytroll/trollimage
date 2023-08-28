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
        cmap_obj = getattr(cmap_module, cmap_var)
        cb = colormap.colorbar(25, 500, cmap_obj)
        channels = [cb[i, :, :] for i in range(3)]
        im = Image(channels=channels, mode="RGB")
        cmap_fn = os.path.join("_static", f"{cmap_var}.png")
        im.save(cmap_fn)

        image = nodes.image(f".. image:: {cmap_fn}", **{"uri": cmap_fn, "alt": cmap_var})
        paragraph = nodes.paragraph(text=cmap_var)
        return [paragraph, image]
