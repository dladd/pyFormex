# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Camera handling menu"""

import pyformex as pf
import draw
import toolbar
from gettext import gettext as _
from guifunc import *

@viewport_function
def zoomIn(*args,**kargs):
        pass
@viewport_function
def zoomOut(*args,**kargs):
        pass
@viewport_function
def dollyIn(*args,**kargs):
        pass
@viewport_function
def dollyOut(*args,**kargs):
        pass
@viewport_function
def panLeft(*args,**kargs):
        pass
@viewport_function
def panRight(*args,**kargs):
        pass
@viewport_function
def panDown(*args,**kargs):
        pass
@viewport_function
def panUp(*args,**kargs):
        pass
@viewport_function
def transLeft(*args,**kargs):
        pass
@viewport_function
def transRight(*args,**kargs):
        pass
@viewport_function
def transDown(*args,**kargs):
        pass
@viewport_function
def transUp(*args,**kargs):
        pass
@viewport_function
def rotLeft(*args,**kargs):
        pass
@viewport_function
def rotRight(*args,**kargs):
        pass
@viewport_function
def rotDown(*args,**kargs):
        pass
@viewport_function
def rotUp(*args,**kargs):
        pass
@viewport_function
def twistRight(*args,**kargs):
        pass
@viewport_function
def twistLeft(*args,**kargs):
        pass
@viewport_function
def lockCamera(*args,**kargs):
        pass
@viewport_function
def unlockCamera(*args,**kargs):
        pass
@viewport_function
def reportCamera(*args,**kargs):
        pass
@viewport_function
def zoomAll(*args,**kargs):
        pass
@viewport_function
def zoomRectangle(*args,**kargs):
        pass


MenuData = [
    (_('&LocalAxes'),draw.setLocalAxes),
    (_('&GlobalAxes'),draw.setGlobalAxes),
    (_('&Projection'),toolbar.setProjection),
    (_('&Perspective'),toolbar.setPerspective),
    (_('&Zoom Rectangle'),zoomRectangle), 
    (_('&Zoom All'),zoomAll), 
    (_('&Zoom In'),zoomIn), 
    (_('&Zoom Out'),zoomOut), 
    (_('&Dolly In'),dollyIn), 
    (_('&Dolly Out'),dollyOut), 
    (_('&Pan Left'),panLeft), 
    (_('&Pan Right'),panRight), 
    (_('&Pan Down'),panDown), 
    (_('&Pan Up'),panUp), 
    (_('&Translate'),[
        (_('Translate &Left'),transLeft), 
        (_('Translate &Right'),transRight), 
        (_('Translate &Down'),transDown),
        (_('Translate &Up'),transUp),
        ]),
    (_('&Rotate'),[
        (_('Rotate &Left'),rotLeft),
        (_('Rotate &Right'),rotRight),
        (_('Rotate &Down'),rotDown), 
        (_('Rotate &Up'),rotUp),
        (_('Rotate &ClockWise'),twistRight),
        (_('Rotate &CCW'),twistLeft),
        ]),
    (_('&Lock'),lockCamera), 
    (_('&Unlock'),unlockCamera), 
    ('---',None),
    (_('&Report'),reportCamera), 
    ]


# End
