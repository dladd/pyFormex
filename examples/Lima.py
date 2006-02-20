#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##

"""Lima examples"""

# We use the lima module
from lima import *
# return standard Turtle rules 
def turtle(rules={}):
    """Return standard Turtle rules, extended and/or overriden by arg.

    The specified arg should be a dictionary of rules, which will extend
    and/or override the default rules.
    """
    d = { 'F' : 'fd();', 'G' : 'fd();', '+' : 'ro(90);', '-' : 'ro(-90);', '*' : 'ro(60);', '/' : 'ro(-60);', 'J':'mv();','K':'mv();', 'X':'', 'Y':'', '[':'push();', ']':'pop();' }
    d.update(rules)
    return d

# 
def extra(rules):
    d = tut
# here are some nice lima generations.
# Each tuple holds an axiom, grow rules, generations and turtle rules
limas = [ [ "F", {"F":"F+G","G":"F-G"},10,turtle() ],
          [ "F", {"F":"F*F//F*F"},6,turtle() ],  # KOCH line
          [ "F+F+F+F", {"F":"FF+FF--FF+F"},5,turtle() ],
          [ "F+F+F+F", {"F":"FF+F+F-F+FF"},5,turtle() ],
          [ "F//F//F", {"F":"F*F//F*F"},5,turtle() ], # KOCH snowflake
          [ "F+F+F+F", {"F":"FF+F++F+F"},4,turtle() ],
          [ "F+F+F+F", {"F":"FF+F+F+F+F+F-F"},4,turtle() ],
          [ "X", {"X":"-YF+XFX+FY-", "Y":"+XF-YFY-FX+"},5,turtle() ], # Hilbert curve
          # more plane filling curves
          [ "F+XF+F+XF", {"X":"XF-F+F-XF+F+XF-F+F-X"},4,turtle() ],
          [ "X", {"X":"XFYFX+F+YFXFY-F-XFYFX", "Y":"YFXFY-F-XFYFX+F+YFXFY"},4,turtle() ],
          [ "XF", {"X":"X*YF**YF/FX//FXFX/YF*", "Y":"/FX*YFYF**YF*FX//FX/Y"},4,turtle() ], # GOSPER line
          [ "F**F**F", {"F":"F*J++F**F", "J":"JJ"},6,turtle() ],
          [ "F*F*F*F*F*F", {"F":"+F/F*F-"},5,turtle({'+':'ro(20);','-':'ro(-20);'}) ],
          [ "F*F*F*F*F*F/F/F/F/F/F*F*F*F*F*F", {"F":"+F/F*F-"},4,turtle({'+':'ro(20);','-':'ro(-20);'}) ],
          # some examples of plants
          [ "+F", {"F":"F[*F]F[/F]F"},5,turtle({'*':'ro(25);','/':'ro(-25);'}) ],
          [ "+Y", {"Y":"YFX[*Y][/Y]", "X":"[/F][*F]FX"},7,turtle({'*':'ro(25);','/':'ro(-25);'}) ],
          [ "+F", {"F":"FF/[/F*F*F]*[*F/F/F]"},4,turtle({'*':'ro(22.5);','/':'ro(-22.5);'}) ],
          # hexagones
          [ "F*F*F*F*F*F", {"F":"[//J*G*F*G]J", "G":"[//K*G*F*G]J"},5,turtle() ],
          [ "F+F", {"F":"F*FF**F**FF*F"},4,turtle() ]
        ]
for a,r,g,t in limas:
    S = [ lima(a,r,i,t) for i in range(g) ]
    # and display them in series
    for s in S:
        if len(s): # avoid error when list is empty
            clear()
            draw(Formex(s))
