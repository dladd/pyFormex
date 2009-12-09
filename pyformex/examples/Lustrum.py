#!/usr/bin/env pyformex --gui
# $Id$
"""Lustrum

level = 'normal'
topics = ['curve','drawing','illustration']
techniques = ['colors','persistence','curve','lima','import']

"""
from pyformex.examples.Lima import *
from project import Project
linewidth(2)
fgcolor(blue)
grow('Plant1',False,False)
P = Project(os.path.join(GD.cfg['datadir'],'blippo-5.pyf'),create=False)
P.load()
curve = P['blippo-5']
draw(curve,color=pyformex_pink,linewidth=5)

## if ask("This is the pyFormex Lustrum release\n\nClick Remove to no longer show this startup screen\n",choices=['Remove','Keep']) == 'Remove':
##     GD.cfg['gui/easter_egg'] = None
    

# End
