# $Id$  *** pyformex ***

"""Texture

Shows how to draw with textures.

level = 'normal'
topics = ['Image','Geometry']
techniques = ['texture']
"""

from gui.imagearray import qimage2numpy
from PyQt4.QtGui import QImage
 

def imagefile2gltexture(filename):
    import Image
    im = Image.open(filename)
    nx,ny = im.size[0],im.size[1]
    try:
        data = im.tostring("raw","RGBA",0,-1)
    except SystemError:
        data = im.tostring("raw","RGBX",0,-1)
    return nx,ny,data


clear()

imagefile = os.path.join(pf.cfg['pyformexdir'],'data','butterfly.png')
imagefile = os.path.join(pf.cfg['pyformexdir'],'data','benedict_6.jpg')
im = QImage(imagefile)
data = qimage2numpy(im,expand=True)
## h,w = im.height(),im.width()
## print im.numBytes()
## buf = im.bits().asstring(im.numBytes())
## data = frombuffer(buf,dtype='ubyte',count=im.numBytes()).reshape(h,w,4)
## data = data[...,[2,1,0,3]] # transform BGRA to RGBA storage
## print data.shape
## print data.dtype

## nx,ny,data = imagefile2gltexture(imagefile)
## print nx,ny,type(data),len(data)
## tex1 = glGenTexture(nx,ny,data)
## print tex1

import simple
F = simple.cuboid()
## draw(F,texture=tex1)

#nx,ny,data = image2gltexture(imagefile)
#print nx,ny,type(data)
## tex2 = glGenTexture(nx,ny,data)
## print tex2

from gui.drawable import Texture
tex3 = Texture(data)
print tex3.tex

draw(F.trl(0,2.),texture=tex3)

#GL.glDeleteTextures(texid)
