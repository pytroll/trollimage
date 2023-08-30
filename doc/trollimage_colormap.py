"""Custom sphinx directive work with colormap objects.

This module must be on the ``sys.path`` and added to the list of extensions
in a ``conf.py``.

"""
import os
import importlib
from typing import Any

from docutils import nodes
from docutils.parsers.rst.directives import flag
from sphinx.util.docutils import SphinxDirective

from trollimage import colormap
from trollimage.image import Image


def setup(app):
    """Add custom extension to sphinx."""
    app.add_directive("trollimage_colormap", TrollimageColormapDirective)

    return {
        "version": "0.1",
        "parallel_read": True,
        "parallel_write": True,
    }


class TrollimageColormapDirective(SphinxDirective):
    """Custom sphinx directive for generating one or more colormap images."""

    required_arguments: int = 1
    option_spec: dict[str, Any] = {
        "category": flag,
    }

    def run(self) -> list[nodes.Node]:
        """Import and generate colormap images to be inserted into the document."""
        cmap_import_path = self.arguments[0]
        is_category = "category" in self.options
        colormap_dict = self._get_colormap_dict(cmap_import_path)

        image_nodes = []
        for cmap_name, cmap_obj in sorted(colormap_dict.items()):
            image_nodes += self._create_colormap_nodes(cmap_name, cmap_obj, is_category)
        return image_nodes

    @staticmethod
    def _get_colormap_dict(cmap_import_path: str) -> dict[str, colormap.Colormap]:
        cmap_module_name, cmap_var = cmap_import_path.rsplit(".", 1)
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
        return cmap_objects

    @staticmethod
    def _create_colormap_nodes(cmap_name: str, cmap_obj: colormap.Colormap, is_category: bool) -> list[nodes.Node]:
        cmap_fn = os.path.join("_static", "colormaps", f"{cmap_name}.png")
        if not os.path.exists(cmap_fn):
            cb = colormap.colorbar(25, 500, cmap_obj, category=is_category)
            channels = [cb[i, :, :] for i in range(3)]
            im = Image(channels=channels, mode="RGB")
            im.save(cmap_fn)

        paragraph = nodes.paragraph(text=cmap_name)
        image = nodes.image("", **{"uri": cmap_fn, "alt": cmap_name})
        return [paragraph, image]
