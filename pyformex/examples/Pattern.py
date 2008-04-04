#!/usr/bin/env pyformex --gui
# $Id: Pattern.py 85 2006-04-02 12:36:40Z bverheg $
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
import simple


if __name__ == "draw":

    #grid = simple.regularGrid([-2,-2],[2,2],[4,4])
    #clear()
    setDrawOptions(dict(view='front',linewidth=5,fgcolor='red'))
    grid = actors.GridActor(nx=(4,4,0),ox=(-2.0,-2.0,0.0),dx=(1.0,1.0,1.0),planes=False,linewidth=1)
    drawActor(grid)
    linewidth(3)
    FA = None
    for n,p in simple.Pattern.items():
        message("%s = %s" % (n,p))
        FB = draw(Formex(pattern(p)),bbox=None,color='red')
        if FA:
            undraw(FA)
        FA = FB

# End
