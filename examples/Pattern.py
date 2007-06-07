#!/usr/bin/env python pyformex.py
# $Id: Pattern.py 85 2006-04-02 12:36:40Z bverheg $

import simple

def drawPattern(p):
    clear()
    F = Formex(pattern(p))
    draw(F,view='front')
    draw(F,view='iso')

for n,p in simple.Pattern.items():
    message("%s = %s" % (n,p))
    drawPattern(p)
