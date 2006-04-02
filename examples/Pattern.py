#!/usr/bin/env pyformex
# $Id$

import simple

def drawPattern(p):
    clear()
    F = Formex(pattern(p))
    draw(F,view='front')
    draw(F,view='iso')

for n,p in simple.Pattern.items():
    message("%s = %s" % (n,p))
    drawPattern(p)
