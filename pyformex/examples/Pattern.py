#!/usr/bin/env pyformex --gui
# $Id: Pattern.py 85 2006-04-02 12:36:40Z bverheg $
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
import simple

def drawPattern(p):
    clear()
    F = Formex(pattern(p))
    draw(F,view='front')
    draw(F,view='iso')

for n,p in simple.Pattern.items():
    message("%s = %s" % (n,p))
    drawPattern(p)
