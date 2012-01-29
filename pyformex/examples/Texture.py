# $Id$  *** pyformex ***

"""Texture

Shows how to draw with textures.

level = 'normal'
topics = ['Image','Geometry']
techniques = ['texture']
"""

from gui.imagearray import image2numpy
from PyQt4.QtGui import QImage

smooth()

clear()

imagefile = os.path.join(pf.cfg['pyformexdir'],'data','butterfly.png')
im = QImage(imagefile)
image = image2numpy(im,expand=True)


import simple
F = simple.cuboid()
#F = Formex('4:0123')#.replic2(3,2)#.toMesh()

draw(F,texture=image,color=white)
view('iso')


# End
