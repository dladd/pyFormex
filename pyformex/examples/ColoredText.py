# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""ColoredText

This example illustrates the drawing of text on the 2D OpenGL canvas.
It also shows how to generate random values and how to remove 2D decorations
from the canvas.

The application starts with the generation of (n,8) random numbers between
0.0 and 1.0, and where n is the number of text strings that will be displayed.
The 8 random numbers for each case are use as follows:

- 0, 1: the relative position on the canvas viewport,
- 2: the relative font size, ranging from 12 to 48,
- 3, 4, 5: the color (resp. red, green and blue components),
- 6: the text to be shown, selected from a list of predefined texts,
- 7: the font to be used, also selected from a list of fonts.

The texts are shown one by one, with a small pause between.
During the display of the first half of the set of texts,
the previous text is removed after each new one is shown.
During the second half, all texts remain on the canvas.
"""
from __future__ import print_function
_status = 'checked'
_level = 'beginner'
_topics = []
_techniques = ['color','text', 'random', 'animation']

from gui.draw import *

def run():
    n = 40
    T = ['Python','NumPy','OpenGL','QT4','pyFormex']
    fonts = ['times','helvetica','fixed']
    ftmin,ftmax = 12,48

    r = random.random((n,8))
    w,h = pf.canvas.width(), pf.canvas.height()
    a = r[:,:2] * array([w,h]).astype(int)
    size = (ftmin + r[:,2] * (ftmax-ftmin)).astype(int)
    colors = r[:,3:6]
    t = (r[:,6] * len(T)).astype(int)
    f = (r[:,7] * len(fonts) - 0.5).astype(int).clip(0,len(fonts)-1)
    clear()

    bgcolor(white)
    lights(False)
    TA = None

    for i in range(n):
        TB = drawText(T[t[i]],a[i][0],a[i][1],font=fonts[f[i]],size=size[i],color=list(colors[i]))
        pause(0.5)
        breakpt()
        if i < n/2:
            undecorate(TA)
        TA = TB

if __name__ == 'draw':
    run()
# End
