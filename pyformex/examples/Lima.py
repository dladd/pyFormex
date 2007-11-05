#!/usr/bin/env python pyformex.py
# $Id: Lima.py 66 2006-02-20 20:08:47Z bverheg $
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
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
limas = {
    'rule0': [ "F", {"F":"F+G","G":"F-G"},10,turtlecmds() ],
    'Koch Line': [ "F", {"F":"F*F//F*F"},6,turtlecmds() ],
    'rule2': [ "F+F+F+F", {"F":"FF+FF--FF+F"},4,turtlecmds() ],
    'rule3': [ "F+F+F+F", {"F":"FF+F+F-F+FF"},4,turtlecmds() ],
    'Koch Snowflake': [ "F//F//F", {"F":"F*F//F*F"},5,turtlecmds() ],
    'rule4': [ "F+F+F+F", {"F":"FF+F++F+F"},4,turtlecmds() ],
    'rule5': [ "F+F+F+F", {"F":"FF+F+F+F+F+F-F"},4,turtlecmds() ],
    'Hilbert Curve': [ "X", {"X":"-YF+XFX+FY-", "Y":"+XF-YFY-FX+"},5,turtlecmds() ],
    'rule7': [ "F+XF+F+XF", {"X":"XF-F+F-XF+F+XF-F+F-X"},4,turtlecmds() ],
    'rule8': [ "X", {"X":"XFYFX+F+YFXFY-F-XFYFX", "Y":"YFXFY-F-XFYFX+F+YFXFY"},4,turtlecmds() ],
    'Gosper Line':      [ "XF", {"X":"X*YF**YF/FX//FXFX/YF*", "Y":"/FX*YFYF**YF*FX//FX/Y"},4,turtlecmds() ],
    'Sierpinski Triangle': [ "F**F**F", {"F":"F*J++F**F", "J":"JJ"},6,turtlecmds() ],
    'rule11': [ "F*F*F*F*F*F", {"F":"+F/F*F-"},5,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ],
    'rule12': [ "F*F*F*F*F*F/F/F/F/F/F*F*F*F*F*F", {"F":"+F/F*F-"},4,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ],
    'plant0': [ "+F", {"F":"F[*F]F[/F]F"},5,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
    'plant1': [ "+Y", {"Y":"YFX[*Y][/Y]", "X":"[/F][*F]FX"},7,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
    'Breezy Bush': [ "+F", {"F":"FF[//F*F*F][*F/F/F]"},4,turtlecmds({'*':'ro(22.55);','/':'ro(-22.5);'}) ],
    'Islands and Lakes': [ "F-F-F-F", {"F":"F-J+FF-F-FF-FJ-FF+J-FF+F+FF+FJ+FFF", "J":"JJJJJJ"},2,turtlecmds() ],
    'Hexagones': [ "F*F*F*F*F*F", {"F":"[//J*G*F*G]J", "G":"[//K*G*F*G]J"},5,turtlecmds() ],
    'Lace': [ "F+F", {"F":"F*FF**F**FF*F"},4,turtlecmds() ],
    'rule19': [ "F++F", {"F":"*F//F*"}, 10, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
    'rule20': [ "F+F+F+F", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
    'rule21': [ "G+G+G+G", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
    #22: [ "+F", {"F":"GH", "G":"GG", "H":"G[*F][/F]"}, 12, turtlecmds({'*':'ro(22.5);','/':'ro(-22.5);'}) ],
    #23: [ "F", {"F":"*F-F*"}, 12, turtlecmds({'*':'ro(45);'}) ],
    #24: [ "JF", {"F":"*F-FF+F*","J":"/J"}, 8, turtlecmds({'*':'ro(45);','/':'ro(-45);'}) ],
    #25: [ "F", {"F":"F-F++F-F"}, 4, turtlecmds() ],
    }

def show(i,L,turtle_cmds):
    """Show the current production of the Lima L."""
    global FA,TA
    turtle_script = L.translate(turtle_cmds)
    coords = turtle.play("reset();" + turtle_script)
    if len(coords) > 0:
        FB = draw(Formex(coords))
        undraw(FA)
        FA = FB
        TB = drawtext("Generation %d"%i,20,20)
        undecorate(TA)
        TA = TB
        

def grow(rule):
    """Show subsequent Lima productions."""
    global FA,TA
    FA = None
    TA = None
    clear()
    GD.message(rule)
    drawtext(rule,20,40)
    a,r,g,t = limas[rule]
    L = lima.Lima(a,r)
    show(0,L,t)
    for i in range(g):
        L.grow()
        show(i+1,L,t)


choices = ['__all__','__custom__'] + limas.keys()

res = askItems([('Production rule',None,'select',choices)])

if res:
    rule = res['Production rule']
    if rule == '__all__':
        for rule in limas.keys():
            grow(rule)
    elif rule == '__all__':
        pass
    else:
        grow(rule)
