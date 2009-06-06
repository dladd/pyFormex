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
sleep(1)

b = a.extrude(2,1.,1)
draw(b,color='red')
sleep(1)

c = b.extrude(4,1.,0)
draw(c,color='blue')
sleep(1)

d = c.extrude(7,1.,2)
draw(d,color='green')
