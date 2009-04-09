#!/usr/bin/env pyformex
# $Id$

"""Extrude

level = 'beginner'
topics = ['geometry']
techniques = ['connect']

"""

clear()
a = Formex([0.,0.,0.])
draw(a,color='black')

b = a.extrude(2,1.,1)
draw(b,color='red')

c = b.extrude(4,1.,0)
draw(c,color='blue')

d = c.extrude(7,1.,2)
draw(d,color='green')
