#!/usr/bin/env python pyformex.py
# $Id: Lima.py 66 2006-02-20 20:08:47Z bverheg $
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Lima examples"""

# We use the lima module
from plugins import lima,turtle

linewidth(2)

# return standard Turtle rules 
def turtlecmds(rules={}):
    """Return standard Turtle rules, extended and/or overriden by arg.

    The specified arg should be a dictionary of rules, which will extend
    and/or override the default rules.
    """
    d = { 'F' : 'fd();', 'G' : 'fd();', '+' : 'ro(90);', '-' : 'ro(-90);', '*' : 'ro(60);', '/' : 'ro(-60);', 'J':'mv();','K':'mv();', 'X':'', 'Y':'', '[':'push();', ']':'pop();' }
    d.update(rules)
    return d
    

# here are some nice lima generations.
# Each tuple holds an axiom, grow rules, generations and turtle rules
limas = [ [ "F", {"F":"F+G","G":"F-G"},10,turtlecmds() ],
          [ "F", {"F":"F*F//F*F"},6,turtlecmds() ],  # KOCH line
          [ "F+F+F+F", {"F":"FF+FF--FF+F"},4,turtlecmds() ],
          [ "F+F+F+F", {"F":"FF+F+F-F+FF"},4,turtlecmds() ],
          [ "F//F//F", {"F":"F*F//F*F"},5,turtlecmds() ], # KOCH snowflake
          [ "F+F+F+F", {"F":"FF+F++F+F"},4,turtlecmds() ],
          [ "F+F+F+F", {"F":"FF+F+F+F+F+F-F"},4,turtlecmds() ],
          [ "X", {"X":"-YF+XFX+FY-", "Y":"+XF-YFY-FX+"},5,turtlecmds() ], # Hilbert curve
          # more plane filling curves
          [ "F+XF+F+XF", {"X":"XF-F+F-XF+F+XF-F+F-X"},4,turtlecmds() ],
          [ "X", {"X":"XFYFX+F+YFXFY-F-XFYFX", "Y":"YFXFY-F-XFYFX+F+YFXFY"},4,turtlecmds() ],
          [ "XF", {"X":"X*YF**YF/FX//FXFX/YF*", "Y":"/FX*YFYF**YF*FX//FX/Y"},4,turtlecmds() ], # GOSPER line
          [ "F**F**F", {"F":"F*J++F**F", "J":"JJ"},6,turtlecmds() ],
          [ "F*F*F*F*F*F", {"F":"+F/F*F-"},5,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ],
          [ "F*F*F*F*F*F/F/F/F/F/F*F*F*F*F*F", {"F":"+F/F*F-"},4,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ],
          # some examples of plants
          [ "+F", {"F":"F[*F]F[/F]F"},5,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
          [ "+Y", {"Y":"YFX[*Y][/Y]", "X":"[/F][*F]FX"},7,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
          [ "+F", {"F":"FF[//F*F*F][*F/F/F]"},4,turtlecmds({'*':'ro(22.55);','/':'ro(-22.5);'}) ], # breezy bush
          # islands and lakes
          [ "F-F-F-F", {"F":"F-J+FF-F-FF-FJ-FF+J-FF+F+FF+FJ+FFF", "J":"JJJJJJ"},2,turtlecmds() ],
          # hexagones
          [ "F*F*F*F*F*F", {"F":"[//J*G*F*G]J", "G":"[//K*G*F*G]J"},5,turtlecmds() ],
          # lace
          [ "F+F", {"F":"F*FF**F**FF*F"},4,turtlecmds() ],
          # Testing new ones
          [ "F++F", {"F":"*F//F*"}, 10, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
          [ "F+F+F+F", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
          [ "G+G+G+G", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
          #[ "+F", {"F":"GH", "G":"GG", "H":"G[*F][/F]"}, 12, turtlecmds({'*':'ro(22.5);','/':'ro(-22.5);'}) ],
          #[ "F", {"F":"*F-F*"}, 12, turtlecmds({'*':'ro(45);'}) ],
          #[ "JF", {"F":"*F-FF+F*","J":"/J"}, 8, turtlecmds({'*':'ro(45);','/':'ro(-45);'}) ],
          #[ "F", {"F":"F-F++F-F"}, 4, turtlecmds() ],
        ]

def show(L,turtle_cmds):
    """Show the current production of the Lima L."""
    turtle_script = L.translate(turtle_cmds)
    coords = turtle.play("reset();" + turtle_script)
    if len(coords) > 0:
        F = Formex(coords)
        clear()
        draw(F)
    breakpt()
    #return F
    
# and display them in series
for a,r,g,t in limas:
    L = lima.Lima(a,r)
    show(L,t)
    for i in range(g):
        L.grow()
        show(L,t)
