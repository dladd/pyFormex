#!/usr/bin/env pyformex --gui
# $Id: Lima.py 66 2006-02-20 20:08:47Z bverheg $
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

level = 'normal'
topics = ['illustration']
techniques = ['dialog','lima']

"""

# We use the lima module
from plugins import lima,turtle
# allow this example to be used as a module
from gui.draw import *

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
    'Dragon Curve': [ "F", {"F":"F+G","G":"F-G"},10,turtlecmds() ],
    'Koch Line': [ "F", {"F":"F*F//F*F"},6,turtlecmds() ],
    'rule2': [ "F+F+F+F", {"F":"FF+FF--FF+F"},4,turtlecmds() ],
    'rule3': [ "F+F+F+F", {"F":"FF+F+F-F+FF"},4,turtlecmds() ],
    'Koch Snowflake': [ "F//F//F", {"F":"F*F//F*F"},5,turtlecmds() ],
    'rule4': [ "F+F+F+F", {"F":"FF+F++F+F"},4,turtlecmds() ],
    'rule5': [ "F+F+F+F", {"F":"FF+F+F+F+F+F-F"},4,turtlecmds() ],
    'Hilbert Curve': [ "X", {"X":"-YF+XFX+FY-", "Y":"+XF-YFY-FX+"},5,turtlecmds() ],
    'Greek Cross Curve': [ "F+XF+F+XF", {"X":"XF-F+F-XF+F+XF-F+F-X"},4,turtlecmds() ],
    'Peano Curve': [ "X", {"X":"XFYFX+F+YFXFY-F-XFYFX", "Y":"YFXFY-F-XFYFX+F+YFXFY"},4,turtlecmds() ],
    'Gosper Curve':      [ "XF", {"X":"X*YF**YF/FX//FXFX/YF*", "Y":"/FX*YFYF**YF*FX//FX/Y"},4,turtlecmds() ],
    'Sierpinski Triangle': [ "F**F**F", {"F":"F*J++F**F", "J":"JJ"},6,turtlecmds() ],
    'Sierpinski Triangle1': [ "F", {"F":"*G/F/G*", "G":"/F*G*F/"},8,turtlecmds() ],
    'Sierpinski Carpet': [ "F+F+F+F", {"F":"JF+F+F+F+JF+F+F+F+J", "J":"JJJ"},3,turtlecmds() ],
    'Gosper Island': [ "F*F*F*F*F*F", {"F":"+F/F*F-"},5,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ],
    'Gosper Island Tiling': [ "F*F*F*F*F*F/F/F/F/F/F*F*F*F*F*F", {"F":"+F/F*F-"},4,turtlecmds({'+':'ro(20);','-':'ro(-20);'}) ],
    'Plant0': [ "+F", {"F":"F[*F]F[/F]F"},5,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
    'Plant1': [ "+Y", {"Y":"YFX[*Y][/Y]", "X":"[/F][*F]FX"},7,turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
    'Breezy Bush': [ "+F", {"F":"FF[//F*F*F][*F/F/F]"},4,turtlecmds({'*':'ro(22.55);','/':'ro(-22.5);'}) ],
    'Islands and Lakes': [ "F-F-F-F", {"F":"F-J+FF-F-FF-FJ-FF+J-FF+F+FF+FJ+FFF", "J":"JJJJJJ"},2,turtlecmds() ],
    'Hexagones': [ "F*F*F*F*F*F", {"F":"[//J*G*F*G]J", "G":"[//K*G*F*G]J"},5,turtlecmds() ],
    'Lace': [ "F+F", {"F":"F*FF**F**FF*F"},4,turtlecmds() ],
    'rule19': [ "F++F", {"F":"*F//F*"}, 10, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
    'rule20': [ "F+F+F+F", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
    'rule21': [ "G+G+G+G", {"F":"*F//G*","G":"/F**G/"}, 8, turtlecmds({'*':'ro(30);','/':'ro(-30);'}) ],
    'Grass': [ "***X", { "F":"FF", "X":"F*[[X]/X]/F[/FX]*X" }, 6, turtlecmds({'*':'ro(25);','/':'ro(-25);'}) ],
    #22: [ "+F", {"F":"GH", "G":"GG", "H":"G[*F][/F]"}, 12, turtlecmds({'*':'ro(22.5);','/':'ro(-22.5);'}) ],
    #23: [ "F", {"F":"*F-F*"}, 12, turtlecmds({'*':'ro(45);'}) ],
    #24: [ "JF", {"F":"*F-FF+F*","J":"/J"}, 8, turtlecmds({'*':'ro(45);','/':'ro(-45);'}) ],
    #25: [ "F", {"F":"F-F++F-F"}, 4, turtlecmds() ],
    }

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
    #print "NGEN default %s" % g
    #print "NGEN requested %s" % ngen
    if ngen >= 0:
        # respect the requested number of generations
        g = ngen
    #print "NGEN executed %s" % g
    L = lima.Lima(a,r)
    # show the axiom
    show(0,L,t,clearing,text)
    # show g generations
    for i in range(g):
        if viewports:
            viewport(i+1)
        L.grow()
        show(i+1,L,t,clearing,text)


def setDefaultGenerations(rule):
    rule = str(rule)
    if limas.has_key(rule):
        ngen = limas[rule][2]
        d = currentDialog()
        if d:
            d.updateData({'ngen':ngen})
        

if __name__ == "draw":

    layout(1)
    viewport(0)
    clear()
    wireframe()
    linewidth(2)
    keys = limas.keys()
    keys.sort()
    choices = ['__all__','__custom__'] + keys

    defaults = {
        'rule':None,
        'ngen':-1,
        'colors':True,
        'clearing':True,
        'viewports':False
        }

    defaults = GD.PF.get('__Lima__data',defaults)

    res = askItems([
        ('rule',defaults['rule'],'select',{'text':'Production rule','choices':choices,'onselect':setDefaultGenerations}),
        ('ngen',defaults['ngen'],{'text':'Number of generations (-1 = default)'}),
        ('colors',defaults['colors'],{'text':'Use different colors per generation'}),
        ('clearing',defaults['clearing'],{'text':'Clear screen between generations'}),
        ('viewports',defaults['viewports'],{'text':'Use a separate viewport for each generation'}),
        ])

    if res:
        globals().update(res)
        GD.PF['__Lima__data'] = res
        if viewports:
            layout(ngen+1,ncols=(ngen+2)//2)
        if rule == '__custom__':
            pass
        elif rule == '__all__':
            for rule in keys:
                res['rule'] = rule
                grow(**res)
        else:
            grow(**res)

# End
