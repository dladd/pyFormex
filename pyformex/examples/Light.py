#!/usr/bin/env pyformex
# $Id$

from gui.actors import *

smooth()
lights(False)

Rendermode = [ 'smooth','flat' ]
Lights = [ False, True ]
Shape = { 'triangle':'16',
          'quad':'123',
          }


color0 = None  # no color: current fgcolor
color1 = red   # single color
color2 = array([red,green,blue]) # 3 colors: will be repeated

## for shape in Shape.keys():
##     F = Formex(mpattern(Shape[shape])).replic2(8,4)
##     color3 = resize(color2,F.shape()) # full color
##     for mode in Rendermode:
##         renderMode(mode)
##         for c in [ color0,color1,color2,color3]:
##             clear()
##             FA = FormexActor(F,color=c)
##             drawActor(FA)
##             zoomAll()
##             for light in Lights:
##                 lights(light)

F = Formex(mpattern(Shape['triangle'])).replic2(8,4)
color3 = resize(color2,F.shape())
renderMode('smooth')
c = color3
clear()
FA = FormexActor(F,color=c)
drawActor(FA)
zoomAll()

def show_changes():
    GD.canvas.setLighting(True)
    GD.canvas.update()
    GD.app.processEvents()
    
text_pos = {'ambient':(None,100),
            'specular':(None,80),
            'emission':(None,60),
            'shininess':(None,40),
            }

def set_light_value(typ,val):
    text,pos = text_pos[typ]
    if text:
        undecorate(text)
    text_pos[typ] = (drawtext('%s = %s'%(typ,val),100,pos),pos)
    setattr(GD.canvas,typ,val)
    show_changes()


def set_ambient(i):
    set_light_value('ambient',i*0.1)

def set_specular(i):
    set_light_value('specular',i*0.1)

def set_emission(i):
    set_light_value('emission',i*0.1)

def set_shininess(i):
    set_light_value('shininess',i*0.1)

nv = 10
vmin = 0.0
vmax = 1.0
dv = (vmax-vmin) / nv
ambi = 10*GD.canvas.ambient
spec = 10*GD.canvas.specular
emis = 10*GD.canvas.emission
shin = 10*GD.canvas.shininess
res = askItems([
    ('ambient',ambi,'slider',{'min':0,'max':10,'func':set_ambient}),
    ('specular',spec,'slider',{'min':0,'max':10,'func':set_specular}),
    ('emission',emis,'slider',{'min':0,'max':10,'func':set_emission}),
    ('shininess',shin,'slider',{'min':0,'max':10,'func':set_shininess}),
    ])

print res
print GD.canvas.ambient
print GD.canvas.specular
print GD.canvas.emission
print GD.canvas.shininess

# End
