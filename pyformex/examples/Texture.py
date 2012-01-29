# $Id$  *** pyformex ***

"""Texture

Shows how to draw with textures.

level = 'normal'
topics = ['Image','Geometry']
techniques = ['texture']
"""

pf.canvas.setBackground(color=['red'])
exit()

from gui.imagearray import image2numpy

clear()
smooth()

imagefile = os.path.join(pf.cfg['pyformexdir'],'data','butterfly.png')
image = image2numpy(imagefile,expand=True)

import simple
F = simple.cuboid()
#F = Formex('4:0123').replic2(3,2).toMesh()
draw(F,texture=image,color=white)
view('iso')

from gui.decors import Rectangle
R = Rectangle(100,100,400,300,color=yellow,texture=image)
decorate(R)

# End
