# $Id$  *** pyformex ***

"""NurbsDecompose

Illustrates some special techniques on Nurbs Curves:

- inserting knots
- curve decomposing
"""
_status = 'checked'
_level = 'advanced'
_topics = ['Geometry', 'Curve']
_techniques = ['nurbs']

from gui.draw import *
from plugins.nurbs import *
from plugins.curve import *
from plugins.nurbs_menu import _options, drawNurbs

class _decors:
    ctrl_numbers = None


def clearDecors():
    undraw(_decors.ctrl_numbers)


def drawNurbs(N):
    clearDecors()
    draw(N,color=_options.color,nolight=True)
    if _options.ctrl:
        draw(N.coords.toCoords(),color=_options.color,nolight=True)
        if _options.ctrl_polygon:
            draw(PolyLine(N.coords.toCoords()),color=_options.color,nolight=True)
        if _options.ctrl_numbers:
            _decors.ctrl_numbers = drawNumbers(N.coords.toCoords())
    if _options.knots:
        draw(N.knotPoints(),color=_options.color,marksize=_options.knotsize)
        if _options.knot_numbers:
            drawNumbers(N.knotPoints())
        if _options.knot_values:
            drawMarks(N.knotPoints(),["%f"%i for i in N.knots],leader='  --> ')


def run():
    clear()
    flat()

    C = Formex('12141214').toCurve()
    #C = Formex('214').toCurve()
    degree = 3

    clear()
    linewidth(1)
    _options.ctrl = True
    _options.ctrl_numbers = True
    _options.ctrl_polygon = True
    _options.knot_values = True

    N = NurbsCurve(C.coords,degree=degree)#,blended=False)
    print N
    _options.linewidth = 1
    _options.color = magenta
    _options.knotsize = 5
    drawNurbs(N)
    zoomAll()

    while True:
        res = askItems([
            dict(name='u',
                 text='New knot values',
                 value='0.2,',
                 )
            ])
        if not res:
            break;

        u = eval('[%s]' % res['u'])
        N = N.insertKnots(u)
        _options.linewidth = 5
        _options.color = blue
        _options.knotsize = 10
        drawNurbs(N)
        zoomAll()

    ## if ack("Remove knots?"):
    ##     u = 0.5
    ##     print N.removeKnots(u,1,0.001)




    if ack("Decompose curve?"):

        N1 = N.decompose()
        print N1
        _options.linewidth = 5
        _options.color = red
        _options.knotsize = 20
        drawNurbs(N1)
        zoomAll()

        ## if ack("Yesho raz?"):
        ##     N2 = N1.decompose()
        ##     _options.linewidth = 5
        ##     _options.color = green
        ##     _options.knotsize = 30
        ##     drawNurbs(N2)
        ##     zoomAll()

        C = BezierSpline(control=N1.coords.toCoords(),degree=N1.degree)
        draw(C,color=blue)


if __name__ == 'draw':
    run()

# End
