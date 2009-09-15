#!/usr/bin/env python
"""Python intro

A short introduction to some aspects of the Python programming language
"""

for light in [ 'green','yellow','red','black',None]:
    if light == 'red':
        print 'stop'
    elif light == 'yellow':
        print 'brake'
    elif light == 'green':
        print 'drive'
    else:
        print 'THE LIGHT IS BROKEN!'



speed_appreciation = { 30:'slow', 60:'normal', 75:'ticket', 90:'dangerous',
                       120:'suicide'}

for speed in 10*range(15):
    

