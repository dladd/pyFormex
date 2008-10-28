#!/usr/bin/env pyformex --gui
"""Script Template

This is a template file to show the general layout of a pyFormex script.
In the current version, a pyFormex script should obey the following rules:
- file name extension is '.py'
- first (comment) line contains 'pyformex'
The script starts by preference with a docstring (like this), composed of a
short first line, a blank line and one or more lines explaining the intention
of the script.
"""
   

def circle(n=60):
    a1 = 360./n
    return Formex([[[cosd(i*a1),sind(i*a1),0.] for i in range(n)]])

def triangle():
    return Formex([[[0.,0.,0.],[1.,0.,0.],[0.5,0.5*sqrt(3.),0.]]])

def square():
    return Formex([[[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]])


class Shape(object):
    def __init__(self,shape,size,position,color):
        self.shape = shape
        self.size = resize(size,(3))
        self.position = position
        self.color = color
        self.F = None
        self.A = None
        self.make()

    def make(self):
        self.F = globals()[self.shape]().scale(self.size).translate(self.position)

    def draw(self):
        self.A = draw(self.F,color=self.color)

    def hide(self):
        if self.A:
            GD.canvas.undraw(self.A)
            self.A = None

    def setSize(size):
        self.size = size
        self.make()

    def setPosition(pos):
        self.position = pos
        self.make()

    def setColor(color):
        self.hide()
        self.color = color
        self.draw()


if __name__ == 'draw':

    smooth()

    A = Shape('circle',10,[110.,80.],'yellow')
    B = Shape('square',[10.,10.],[30.,30.],'white')
    C = Shape('square',[80.,60.],[10.,0.],'red')
    D = Shape('triangle',[100.,40.],[0.,60.],'green')
    A.draw()
    B.draw()
    C.draw()
    D.draw()
    zoomall()
# End
