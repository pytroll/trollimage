"""Simple utilities functions to handle colormaps and other use cases."""

from __future__ import division
import numpy as np
import sys
from trollimage.colormap import Colormap
from trollimage.image import Image
from PIL import Image as Pimage

def _hex_to_rgb(value):
    '''
    _hex_to_rgb converts a string of 3 hex color values
    into a tuple of decimal values
    '''
    value = value.lstrip('#')
    dec = int(value, 16)
    return dec 

def _text_to_rgb(value,norm=False,cat=1, tot=1,offset=0.5,hex=False):
    '''
    _text_to_rgb takes as input a string composed by 3 values in the range [0,255]
    and returns a tuple of integers. If the parameters cat and tot are given,
    the function generates a transparency value for this color and returns a tuple
    of length 4.
    tot is the total number of colors in the colormap
    cat is the index of the current colour in the colormap
    if norm is set to True, the input values are normalized between 0 and 1.
    '''
    tokens = value.split()
    if hex:
            for i in range(len(tokens)):
                tokens[i] = _hex_to_rgb(tokens[i]) 
    transparency = float(cat)/float(tot)+offset
    if transparency > 1.0:
        transparency = 1.0
    if norm:
        return (float(tokens[0])/255.0, float(tokens[1])/255.0, float(tokens[2])/255.0, transparency)
    else:    
        return (int(tokens[0]), int(tokens[1]), int(tokens[2]), int(round(transparency * 255.0)))


def _make_cmap(colors, position=None, bit=False):
    '''
    _make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). _make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    palette = [(i, (float(r), float(g), float(b), float(a))) for
    i, (r, g, b, a) in enumerate(colors)]
    cmap = Colormap(*palette)
    return cmap


def cmap_from_text(filename, norm=False, transparency=False, hex=False):
    '''
    cmap_from_text takes as input a file that contains a colormap in text format
    composed by lines with 3 values in the range [0,255] or [00,FF]
    and returns a tuple of integers. If the parameters cat and tot are given,
    the function generates a transparency value for this color and returns a tuple
    of length 4.
    tot is the total number of colors in the colormap
    cat is the index of the current colour in the colormap
    if norm is set to True, the input values are normalized between 0 and 1.
    '''
    lines = [line.rstrip('\n') for line in open(filename)] 
    _colors=[]
    _tot = len(lines)
    _index = 1
    for i in lines:
        if transparency:
            _colors.append(_text_to_rgb(i,norm=norm,cat=_index,tot=_tot,hex=hex))
        else:
            _colors.append(_text_to_rgb(i,norm=norm,hex=hex))
        _index = _index + 1
    return _make_cmap(_colors)


def _image2array(filepath):
    '''
    Utility function that converts an image file in 3 np arrays
    that can be fed into geo_image.GeoImage in order to generate
    a PyTROLL GeoImage object.
    '''
    im = Pimage.open(filepath).convert('RGB')
    (width, height) = im.size
    _r = np.array(list(im.getdata(0)))/255.0
    _g = np.array(list(im.getdata(1)))/255.0
    _b = np.array(list(im.getdata(2)))/255.0
    _r = _r.reshape((height, width))
    _g = _g.reshape((height, width))
    _b = _b.reshape((height, width))
    return _r, _g, _b

def pilimage2trollimage(pimage):
    (r,g,b) = _image2array(pimage)
    return Image((r,g,b), mode="RGB")
