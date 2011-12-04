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



appreciation = { 0: 'not driving', 30:'slow', 60:'normal', 90:'dangerous', 120:'suicidal'}

for i in range(5):
    speed = 30*i
    print "%s. Driving at speed %s is %s" % (i,speed,appreciation[speed])


# End
