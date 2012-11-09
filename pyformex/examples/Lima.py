# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Lima examples

This example illustrates the use of the lima and turtle plugins.

lima
----
The lima plugin allows the generation of fractal-like structures through
the use of a Lindenmayer formalism.

turtle
------
The turtle plugin allows the creation of line drawings through Turtle
graphics.
"""
from __future__ import print_function

_status = 'checked'
_level = 'normal'
_topics = ['illustration']
_techniques = ['dialog','lima']

from gui.draw import *
from plugins import lima,turtle
from odict import ODict

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
limas = ODict([
    ('Dragon Curve', [ "F", {"F":"F+G","G":"F-G"},10,turtlecmds() ]),
    ('Koch Line', [ "F", {"F":"F*F//F*F"},6,turtlecmds() ]),
    ('rule2', [ "F+F+F+F", {"F":"FF+FF--FF+F"},4,turtlecmds() ]),
    ('rule3', [ "F+F+F+F", {"F":"FF+F+F-F+FF"},4,turtlecmds() ]),
    ('Koch Snowflake', [ "F//F//F", {"F":"F*F//F*F"},5,turtlecmds() ]),
    ('rule4', [ "F+F+F+F", {"F":"FF+F++F+F"},4,turtlecmds() ]),
    ('rule5', [ "F+F+F+F", {"F":"FF+F+F+F+F+F-F"},4,turtlecmds() ]),
    ('Hilbert Curve', [ "X", {"X":"-YF+XFX+FY-", "Y":"+XF-YFY-FX+"},5,turtlecmds() ]),
    ('Greek Cross Curve', [ "F+XF+F+XF", {"X":"XF-F+F-XF+F+XF-F+F-X"},4,turtlecmds() ]),
    ('Peano Curve', [ "X", {"X":"XFYFX+F+YFXFY-F-XFYFX", "Y":"YFXFY-F-XFYFX+F+YFXFY"},4,turtlecmds() ]),
    ('Gosper Curve',      [ "XF", {"X":"X*YF**YF/FX//FXFX/YF*", "Y":"/FX*YFYF**YF*FX//FX/Y"},4,turtlecmds() ]),
    ('Sierpinski Triangle', [ "F**F**F", {"F":"F*J++F**F", "J":"JJ"},6,turtlecmds() ]),
    ('Sierpinski Triangle1', [ "F", {"F":"*G/F/G*", "G":"/F*G*F/"},8,turtlecmds() ]),
    ('Sierpinski Carpet', [ "F+F+F+F", {"F":"JF+F+F+F+JF+F+F+F+J", "J":"JJJ"},3,turtlecmds() ]),
    ('Gosper Island', [ "F*F*F*F*F*F", {"F":"+F/F*F-"},5,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ]),
    ('Gosper Island Tiling', [ "F*F*F*F*F*F/F/F/F/F/F*F*F*F*F*F", {"F":"+F/F*F-"},4,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ]),
    ('Plant0', [ "+F", {"F":"F[*F]F[/F]F"},5,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ]),
    ('Plant1', [ "+Y", {"Y":"YFX[*Y][/Y]", "X":"[/F][*F]FX"},7,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ]),
    ('Breezy Bush', [ "+F", {"F":"FF[//F*F*F][*F/F/F]"},4,turtlecmds({'*':'ro(22.55);','/':'ro(-22.5);'}) ]),
    ('Islands and Lakes', [ "F-F-F-F", {"F":"F-J+FF-F-FF-FJ-FF+J-FF+F+FF+FJ+FFF", "J":"JJJJJJ"},2,turtlecmds() ]),
    ('Hexagones', [ "F*F*F*F*F*F", {"F":"[//J*G*F*G]J", "G":"[//K*G*F*G]J"},5,turtlecmds() ]),
    ('Lace', [ "F+F", {"F":"F*FF**F**FF*F"},4,turtlecmds() ]),
    ('rule19', [ "F++F", {"F":"*F//F*"}, 10, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ]),
    ('rule20', [ "F+F+F+F", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ]),
    ('rule21', [ "G+G+G+G", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ]),
    ('Grass', [ "***X", { "F":"FF", "X":"F*[[X]/X]/F[/FX]*X" }, 6, turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ]),
    # (rule22', [ "+F", {"F":"GH", "G":"GG", "H":"G[*F][/F]"}, 12, turtlecmds({'*':'ro(22.5);','/':'ro(-22.5);'}) ]),
    # (rule23', [ "F", {"F":"*F-F*"}, 12, turtlecmds({'*':'ro(45);'}) ]),
    # (rule24', [ "JF", {"F":"*F-FF+F*","J":"/J"}, 8, turtlecmds({'*':'ro(45);','/':'ro(-45);'}) ]),
    # (rule25', [ "F", {"F":"F-F++F-F"}, 4, turtlecmds() ]),
    ])

def show(i,L,turtle_cmds,clear=True,text=True,colors=True):
    """Show the current production of the Lima L."""
    global FA,TA
    turtle_script = L.translate(turtle_cmds)
    coords = turtle.play("reset();" + turtle_script)
    if len(coords) > 0:
        if colors:
            prop = i
        else:
            prop = 0
        FB = draw(Formex(coords,prop))
        if clear:
            undraw(FA)
        FA = FB
        if text:
            TB = drawText("Generation %d"%i,40,40,size=24)
            undecorate(TA)
            TA = TB
        

def grow(rule='',clearing=True,text=True,ngen=-1,colors=True,viewports=False):
    """Show subsequent Lima productions."""
    global FA,TA
    FA = None
    TA = None
    viewport(0)
    clear()
    if not rule in limas.keys():
        return
    
    if text:
        drawText(rule,40,60,size=24)

    a,r,g,t = limas[rule]
    if ngen >= 0:
        # respect the requested number of generations
        g = ngen
        
    if viewports:
        layout(g+1,ncols=(g+2)//2)

    L = lima.Lima(a,r)
    # show the axiom
    show(0,L,t,clearing,text)
    # show g generations
    for i in range(g):
        if viewports:
            viewport(i+1)
            clear()
        L.grow()
        show(i+1,L,t,clearing,text)


def setDefaultGenerations(rule):
    rule = str(rule)
    if rule in limas:
        ngen = limas[rule][2]
        d = currentDialog()
        if d:
            d.updateData({'ngen':ngen})
        

def run():
    layout(1)
    viewport(0)
    clear()
    wireframe()
    linewidth(2)
    keys = limas.keys()
    #keys.sort()
    choices = ['__all__','__custom__'] + keys

    defaults = {
        'rule':None,
        'ngen':-1,
        'colors':True,
        'clearing':True,
        'viewports':False
        }

    defaults = pf.PF.get('__Lima__data',defaults)

    res = askItems([
        dict(name='rule',value=defaults['rule'],text='Production rule',choices=choices,onselect=setDefaultGenerations),
        dict(name='ngen',value=defaults['ngen'],text='Number of generations (-1 = default)'),
        dict(name='colors',value=defaults['colors'],text='Use different colors per generation'),
        dict(name='clearing',value=defaults['clearing'],text='Clear screen between generations'),
        dict(name='viewports',value=defaults['viewports'],text='Use a separate viewport for each generation'),
        ])

    if res:
        pf.PF['__Lima__data'] = res.copy()
        rule = res['rule']
        if rule == '__custom__':
            pass
        elif rule == '__all__':
            for r in keys:
                res['rule'] = r
                grow(**res)
            res['rule'] = rule # reset !
        else:
            grow(**res)

if __name__ == 'draw':
    run()
# End
