# $Id$  *** pyformex ***

"""Texture

Shows how to draw with textures and how to set a background image.

level = 'normal'
topics = ['Image','Geometry']
techniques = ['texture']

"""
from gui.draw import *
from gui.imagearray import image2numpy

def run():
    clear()
    smooth()

    imagefile = os.path.join(pf.cfg['pyformexdir'],'data','butterfly.png')
    image = image2numpy(imagefile,expand=True)

    import simple
    F = simple.cuboid().centered()
    G = Formex('4:0123').replic2(3,2).toMesh().setProp(range(1,7)).centered()
    draw([F,G],texture=image)
    view('iso')
    zoom(0.5)

    from gui.decors import Rectangle
    R = Rectangle(100,100,400,300,color=yellow,texture=image)
    decorate(R)

    bgcolor(color=white,image=imagefile)

if __name__ == 'draw':
    run()
# End
