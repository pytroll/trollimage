from trollimage import utilities as tu


#
#  Examples: importing colormaps 
#
filename='setvak.rgb'
my_cmap = tu.cmap_from_text(filename)
print(my_cmap.colors)
my_cmap_norm = tu.cmap_from_text(filename, norm=True)
print(my_cmap_norm.colors)
my_cmap_transp = tu.cmap_from_text(filename,norm=True, transparency=True)
print(my_cmap_transp.colors)
filename='hrv.rgb'
my_cmap_hex = tu.cmap_from_text(filename,hex=True)
print(my_cmap_hex.colors)

#
#  Example: converting PIL to trollimage.Image
#
image = "pifn.png"
timage = tu.pilimage2trollimage(image)
