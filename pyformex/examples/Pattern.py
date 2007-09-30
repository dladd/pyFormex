#!/usr/bin/env python pyformex.py
# $Id: Pattern.py 85 2006-04-02 12:36:40Z bverheg $
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
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
