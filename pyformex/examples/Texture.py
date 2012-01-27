# $Id$  *** pyformex ***

from gui.imagearray import qimage2numpy
from PyQt4.QtGui import QImage

def image2gltexture(im,flip=True):
    """Convert a bitmap image to texture data.

    """
    data = qimage2numpy(QImage(im),expand=True)
    if flip:
        data = flipud(data)
    print data.dtype
    print data.shape
    print data['r'].shape
    data = dstack([data['r'],data['g'],data['b'],data['a']]).reshape(data.shape+(4,))
    print data.dtype
    print data.shape
    data = require(data,dtype='ubyte',requirements='C')
    ny,nx = data.shape[:2]
    return nx,ny,data
 

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

## nx,ny,data = imagefile2gltexture(imagefile)
## print nx,ny,type(data),len(data)
## tex1 = glGenTexture(nx,ny,data)
## print tex1

import simple
F = simple.cuboid()
## draw(F,texture=tex1)

nx,ny,data = image2gltexture(imagefile)
print nx,ny,type(data)
## tex2 = glGenTexture(nx,ny,data)
## print tex2

from gui.drawable import Texture
tex3 = Texture(data)
print tex3.tex

draw(F.trl(0,2.),texture=tex3)

#GL.glDeleteTextures(texid)
