#!/usr/bin/env pyformex
# $Id$
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

"""Optical Illusions

level = 'normal'
topics = ['illustration','geometry']
techniques = ['dialog', 'draw', 'persistence','random']
acknowledgements = ['Tomas Praet']

"""

from simple import *
from gui import widgets
from gui.widgets import simpleInputItem as I
from odict import ODict

################# Illusion definitions #####################

def ParallelLines():
    """Parallel Lines

    This illustration consists only of equally sized squares.
    Though the thus formed horizontal lines seem to converge,
    they are strictly parallel.
    """
    resetview([0.8,0.8,0.8])
    lines = Formex([[[0,0,0],[14,0,0]]]).replic(13,1,dir=1)
    draw(lines,color=[0.8,0.8,0.8],linewidth=1.0)
    F = Formex(mpattern('1234')).replic(13,1)
    F.setProp([0,7])
    F += F.translate([0.2,1,0]) + F.translate([0.4,2,0]) + F.translate([0.2,3,0])
    F = F.replic(3,4,dir=1)
    draw(F)


def RotatingCircle():
    """Rotating Circle

    When staring at the cross in the middle,
    the disappearing magenta circles create the illusion of a green
    rotating circle. If you keep concentrating your gaze on the centre,
    the green circle wil seem to devour the magenta circle, up to a point
    where you no longer see the magenta circles.
    Blinking or changing your focus will immediately undo the effect.
    """
    resetview([0.8,0.8,0.8])
    ask = askItems([('Number of circles',12),('Radius of circles',1.2),('Radius of the figure',12),('Number of rotations',16),('color of circles',[1.0,0.40,1.0]),('Sleep time',0.03),('Zoom',14.)])
    if not ask: return
    N = ask['Number of circles']
    r = ask['Radius of circles']
    R = ask['Radius of the figure']
    n = ask['Number of rotations']
    col = ask['color of circles']
    sl = ask['Sleep time']
    sc = ask['Zoom']
    box=[[-sc,-sc,-sc],[sc,sc,sc]]
    draw(shape('plus'),bbox=box)
    C = circle(a1=20).scale([r,r,0]).points()
    O = [0,0,0]
    F = Formex([[C[i,0],C[i+1,0],O] for i in arange(0,36,2)]).translate([R,0,0]).rosette(N-1,360./N)
    for i in range(n*N):
        F = F.rotate(-360./N)
        dr = draw(F,color=col,bbox=box)
        if i>0: undraw(DR)
        DR=dr
        sleep(sl)


def SquaresAndCircles():
    """Squares And Circles

    When you look at one of the white circles, the other circles seem to
    change color. In fact, the color they appear to have is the same as
    the color of the squares. This color is generated randomly.
    Press 'show' multiple times to see the effect of the color of the squares.
    You may need to zoom to get an optimal effect.
    """
    resetview([0.6,0.6,0.6])
    B,H = 16,16
    F = Formex(mpattern('1234')).replic2(B,H,1.2,1.2)
    R = 0.2/sqrt(2.)
    C = circle(a1=20).scale(R).points()
    O = [0,0,0]
    G = Formex([[C[i,0],C[i+1,0],O] for i in arange(0,36,2)]).translate([1.1,1.1,0]).replic2(B-1,H-1,1.2,1.2)
    G.setProp(7)
    draw(F,color=random.rand(3)/2)
    draw(G)


def ShadesOfGrey():
    """Shades Of Grey

    Our perception of brightness is relative.
    Therefore, the figure on the left looks a little darker than the one right.
    The effect can be somewhat subtle though.
    """
    resetview([0.8,0.8,0.8])
    sc = 2
    box = [[-2,0,-2],[2,8,2]]
    back = Formex(mpattern('1234')).scale([8,8,1])
    back += back.translate([-8,0,0])
    back.setProp([0,7])
    C = circle(a1=11.25).rotate(-90,2).points()
    F = Formex([[C[i,0],C[i+1,0],2*C[i+1,0],2*C[i,0]] for i in range(0,32,2)]).translate([2,4,0])
    n = 40
    for i in range(n):
        F = F.translate([-2./n,0,0])
        G = F.reflect(0)
        dr1 = draw(F+G,color=[0.6,0.6,0.6],bbox=box)
        dr2 = draw(back,bbox=box)
        if i>0:
            undraw(DR2)
            undraw(DR1)
        else:
            sleep(2)
        DR1 = dr1
        DR2 = dr2


def RunningInCircles():
    """Running In Circles

    If you don't look directly at the rectangles, both rectangles will
    appear to 'overtake' each other constantly, although they are moving
    at equal and constant speed.
    """
    resetview()
    box= [[-8,-8,-8],[8,8,8]]
    N = 72
    R = 10
    C = circle(a1=360./N).points()
    O =[0,0,0]
    F = Formex([[C[i,0],C[i+1,0],O] for i in arange(0,2*N,2)]).scale([R,R,0])
    F.setProp([0,7])
    p = circle(a1=360./N).vertices()
    centre = Formex([add(p[0:len(p):2],p[-1])]).translate([-1,0,0])
    centre.setProp(1)
    draw(centre,bbox=box)
    draw(F,bbox=box)
    b1 = Formex(mpattern('1234')).scale([1.5,0.8,0]).translate([0,8.5,0.1])
    b1.setProp(3)
    b2 = Formex(mpattern('1234')).scale([1.5,0.8,0]).translate([0,7,0.1])
    b2.setProp(6)
    b = b1+b2
    col = [random.rand(3)/3,[1,1,1]-random.rand(3)/8]
    for i in range(4*N):
        b = b.rotate(360./N/4)
        dr = draw(b,bbox=box,color=col)
        if i>0:
            undraw(DR)
        DR = dr


def HowManyColors():
    """How Many Colors

    How many colors are there in this image?
    It looks like there are 4 colors (pink, orange, light green and cyan),
    but in fact, there are only 3. The blueish and greenish colors are
    exactly the same.
    Lots of zooming might convince you that this is the case.
    """
    resetview()
    magenta,orange,cyan = array([1.,0.,1.]),array([1.,0.6,0.]),array([0.,1.,0.6])
    b,h,B,H = 10,0.5,11,99
    F = Formex(mpattern('1234')).scale([b,h,1]).replic2(B,H,b,h)
    col = resize(magenta,(H,B,3))
    for i in range(H):
        for j in range(B):
            if i%2==0:
                if j%4==1: col[i,j]=cyan
            else:
                if j%4==3: col[i,j]=cyan
                else: col[i,j]=orange
    draw(F,color=col.reshape(-1,3))


def AlignedLines():
    """Aligned Lines

    This is a classic optical illusion.
    Straight lines can appear to be shifted when only a tilted part is
    visible.
    """
    resetview()
    a = 60.
    lines = Formex(pattern('1')).scale([20,1,0]).rotate(a).translate([-20.*cos(a*pi/180.),0,0]).replic(32,1)
    lines = lines.cutWithPlane([-1,0,0],[1,0,0],side='+').cutWithPlane([22,0,0],[1,0,0],side='-')
    mask = Formex(mpattern('1234')).scale([1,20.*sin(a*pi/180.),1]).replic(11,2)
    mask.setProp(6)
    savedelay = pf.GUI.drawwait
    draw(mask,color=random.rand(3))
    delay(2)
    draw(lines,linewidth=2)
    for i in range(3):
        wait()
        renderMode('wireframe')
        wait()
        renderMode('flat')
    delay(savedelay)


def ParallelLinesOverWheel():
    """Parallel Lines Over Wheel

    Another commonly seen illusion.
    The lines in the back tend to give the illusion of curved lined,
    although they are completely straight.
    Some zooming can help to optimize the effect.
    """
    resetview()
    C,O = circle(a1=20).scale(2).points(),[0,0,0]
    draw(Formex([[C[i,0],C[i+1,0],O] for i in arange(0,36,2)]))
    line = Formex([[[-20,0,0],[20,0,0]]])
    lines = line
    hor = line.translate([0,-4,0]) + line.translate([0,4,0])
    draw(hor,color=red,linewidth=4)
    for i in range(0,180,5):
        lines += line.rotate(i)
    draw(lines,linewidth=1)


def MotionInducedBlindness():
    """Motion Induced Blindness

    This is a very nice illusion.
    Look at the centre of the image.
    The moving background will give the illusion that the other static points
    disappear.
    Blinking or changing your focus will immediately undo the effect.
    Cool huh?
    """
    resetview('black')
    res = askItems([('Number of static points',10),('Background',None,'radio',{'choices':['Tiles','Structured points','Random points']}),('Rotations',2),('Rotation angle',2),('Number of random points',300)])
    if not res: return
    nr,a,rot,back,n = res['Number of random points'],res['Rotation angle'],res['Rotations'],res['Background'],res['Number of static points']
    draw(shape('star').scale(0.4),color=red,linewidth=2)
    points = Formex([[0,-10,0]]).rosette(n,360./n)
    draw(points,color=random.rand(3),marksize=10)
    col=random.rand(3)
    if back=='Tiles': F = shape('plus').replic2(11,11,3,3).translate([-15,-15,0])
    elif back=='Structured points': F = Formex([[0,0,0]]).replic2(30,30,1).translate([-15,-15,0])
    else: F = Formex(random.rand((nr,3))).scale([30,30,0]).translate([-15,-15,0])
    for i in range(rot*360/a):
        F = F.rotate(a)
        dr = draw(F,color=col,linewidth=2,bbox=[[-10,-10,0],[10,10,0]])
        if i>0: undraw(DR)
        DR = dr


def FlickerInducedBlindness():
    """Flicker Induced Blindness

    """
    #... STILL UNDER DEVELOPMENT... (pyFormex might lack the possibility to reach the correct frequencies)
    resetview('black')
    n,freq,d = 4,2.,0.17
    sl = 1/2/freq
    centre = shape('plus').scale(0.4)
    centre.setProp(7)
    draw(centre)
    points = Formex([[0,-5,0]]).rosette(n,360./n)
    points.setProp(6)
    draw(points,color=[0.2,0.5,0.3],marksize=4)
    F1 = Formex(mpattern('1234')).scale([1,2,1]).translate([d,-6,0])
    F2 = F1.translate([-1-2*d,0,0])
    F1 = F1.rosette(n,360./n)
    F2 = F2.rosette(n,360./n)
    for i in range(200):
        dr1 = draw(F1,color=[0.4,0.,0.])
        if i>0: undraw(dr2)
        sleep(sl)
        dr2 = draw(F2,color=[0.4,0.,0.])
        undraw(dr1)
        sleep(sl)


def SineWave():
    """Sine Wave

    A simple yet powerful illusion: the vertical lines seem larger at places
    where the sine wave is more horizontal, and smaller where the sine wave
    is more vertical.
    
    Play with the parameters to get a more obvious result. Amplitude 0 shows
    that all lines are equally large.
    """
    resetview()
    res = askItems([('Amplitude',3.5),('Periods',3),('Spacing between lines',0.1)])
    if not res: return
    shift,per,amp = res['Spacing between lines'],res['Periods'],res['Amplitude']
    n = int(2*pi/shift*per)
    F = Formex(mpattern('2')).replic(n,shift)
    for i in F:
        i[0,1] = amp*sin(i[0,0])
        i[1,1] = i[0,1] + 1
    draw(F)


def CirclesAndLines():
    """Circles And Lines

    Another classic. All lines are completely straight, though the circles
    in the background give you the illusion that they aren't.
    The colors are generated randomly; some combination might not work as
    well as others.
    """
    resetview()
    n,m,nc = 5,5,8
    size = nc*sqrt(2)
    lines = Formex(pattern('1234')).translate([-0.5,-0.5,0]).rotate(45).scale(size)
    c = circle(a1=5)
    C = c
    for i in range(2,nc+1): C += c.scale(i)
    C = C.replic2(n,m,2*nc,2*nc)
    lines = lines.replic2(n,m,2*nc,2*nc)
    draw(C,linewidth=2,color=random.rand(3))
    draw(lines,linewidth=3,color=random.rand(3))


def Crater():
    """Crater

    Though this image is 2D, you might get a 3D illusion.
    Look carefully and you'll see the whirlabout of a crater shaped object.
    """
    resetview()
    deg,rot,col = 5,3,random.rand(2,3)
    r1,r2,r3,r4,r5,r6,r7,r8 = 1,1.9,2.9,4,5.1,6.1,7,7.8
    p = circle(a1=5).vertices()
    C = Formex([p[0:len(p):2]])
    C1 = C.scale(r1).translate([0,r1,0])
    C2 = C.scale(r2).translate([0,r2,0])
    C3 = C.scale(r3).translate([0,r3,0])
    C4 = C.scale(r4).translate([0,r4,0])
    C5 = C.scale(r5).translate([0,2*r4-r5,0])
    C6 = C.scale(r6).translate([0,2*r4-r6,0])
    C7 = C.scale(r7).translate([0,2*r4-r7,0])
    C8 = C.scale(r8).translate([0,2*r4-r8,0])
    fig = C1+C2+C3+C4+C5+C6+C7+C8
    for i in range(rot*360/deg):
        fig = fig.rotate(deg)
        dr = draw(fig,color=col)
        #dr = draw(fig,color=[[0.8,0.8,0.8],[0.2,0.2,0.2]])
        if i>0: undraw(DR)
        DR = dr


def Cussion():
    """Cussion

    This is a powerful illusion, though again some color combinations might
    not work as well as others.
    The smaller squares on this 'chessboard' tend to give a distortion.
    Again, all horizontal and vertical lines are perfectly parallel and
    straight!
    """
    resetview()
    b,h = 17,17
    if b%2==0: b+=1
    if h%2==0: h+=1
    chess = Formex(mpattern('1234')).replic2(b,h,1,1).translate([-b/2+0.5,-h/2+0.5,0])
    col=[random.rand(3),random.rand(3)]
    sq1 = Formex(mpattern('1234')).scale([0.25,0.25,1]).translate([-0.45,0.2,0])
    sq2 = Formex(mpattern('1234')).scale([0.25,0.25,1]).translate([0.2,-0.45,0])
    F = sq1.translate([1,0,0]).replic(int(b/2)-1,1)+sq2.translate([0,1,0]).replic(int(h/2)-1,1,dir=1)
    sq = sq1+sq2
    for i in range(int(b/2)):
        for j in range(int(h/2)):
            if i+j < (int(b/2)+int(b/2))/2-1: F += sq.translate([i+1,j+1,0])
    colors=ndarray([0,0])
    for i in F:
        if (int(i[0,0])+int(i[0,1])-1)%2 == 0: colors=append(colors,col[1])
        else: colors=append(colors,col[0])
    colors= colors.reshape(-1,3)
    F = F.rosette(4,90)
    draw(F,color=colors.reshape(-1,3))
    draw(chess,color=col)


def CrazyCircles():
    """Crazy Circles

    You've undoubtably seen some variation of this illusion before.
    Looking at different spots of the image will give the illusion of motion.
    The secret is all in the combination of colors. Zooming might increase
    the effect.
    Switching to wireframe removes the effect and proves that this is merely
    an illusion.
    """
    resetview()
    n = 5*6
    col = [[1.,1.,1.],[0.12,0.556,1.],[0.,0.,1.],[0.,0.,0.],[0.7,0.9,0.2],[1.,1.,0.]]
    p = circle(a1=10).rotate(5).vertices()
    f = Formex([p[0:len(p):2]])
    F = f.copy()
    for i in range(1,n):
        F += f.scale([1+i*0.5,1+i*0.5,1])
    F1 = F.replic2(5,5,n+1,n+1)
    F2 = F.replic2(4,4,n+1,n+1).translate([(n+1)/2,(n+1)/2,0])
    draw(F1+F2,color=col)


############ Other actions #################
    
def resetview(bgcol='white'):
    clear()
    reset()
    layout(nvps=1)
    bgcolor(bgcol)
    renderMode('flat')
    toolbar.setProjection()
    frontView()


############# Create dialog #################

gdname = '_OpticalIllusions_data_'
dialog = None
explanation = None

illusions = [
    ParallelLines,
    RotatingCircle,
    SquaresAndCircles,
    ShadesOfGrey,
    RunningInCircles,
    HowManyColors,
    AlignedLines,
    ParallelLinesOverWheel,
    MotionInducedBlindness,
    ## FlickerInducedBlindness,
    SineWave,
    CirclesAndLines,
    Crater,
    Cussion,
    CrazyCircles,
    ]

headers = [ getattr(f,'__doc__').split('\n')[0] for f in illusions ]
method = ODict(zip(headers,illusions))

# Dialog Actions

def close():
    """Close the dialog"""
    global dialog,explanation
    if dialog:
        pf.PF[gdname] = dialog.results
        dialog.close()
        dialog = None
    if explanation:
        explanation.close()
        explanation = None
    

def explain():
    """Show the explanation"""
    global explanation
    dialog.acceptData()
    globals().update(dialog.results)
    text = method[Illusion].__doc__
    if Explain:
        # use a persistent text box
        if explanation:
            explanation.updateText(text)
            explanation.raise_()
        else:
            # create the persistent text box
            explanation = widgets.TextBox(text,actions=[('Close',None)])
            explanation.show()
    else:
        # show a non-persistent text
        showText(method[Illusion].__doc__)

def show():
    """Show the illusion"""
    dialog.acceptData()
    globals().update(dialog.results)
    if Explain:
        explain()
    dialog.hide()
    method[Illusion]()
    dialog.show()

def next():
    """Show the next illusion"""
    dialog.acceptData()
    ill = dialog.results['Illusion']
    print ill,method._order
    i = (method._order.index(ill) + 1) % len(method._order)
    print method._order[i]
    dialog.updateData({'Illusion':method._order[i]})
    show()

def timeOut():
    """What to do when the dialog receives a timeout signal"""
    show()
    close()


def openDialog():
    """Create and display the dialog"""
    global dialog,explanation
    data_items = [
        I('Illusion',Illusion,choices=method.keys()),
        I('Explain',Explain,text='Show explanation'),
        ]

    dialog = widgets.NewInputDialog(
        data_items,
        caption='Optical illusions',
        actions = [('Done',close),
                   ('Next',next),
                   ('Explain',explain),
                   ('Show',show)],
        default='Show'
        )
    dialog.timeout = timeOut
    dialog.show()
    

if __name__ == "draw":
    try:
        globals().update(pf.PF[gdname])
    except:
        Illusion = None
        Explain = False


    print Illusion,Explain
    close()
    openDialog()
    
# End
