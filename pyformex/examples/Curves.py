#!/usr/bin/env pyformex --gui
# $Id$

from plugins.curve import *

"""Curves

Examples showing the use of the 'curve' plugin

level = 'normal'
topics = ['geometry','curves']
techniques = ['spline','solve']
"""


def SelectPoint(P,s, atol=1.e-5, rtol=1.e-5):
    """take a Formex-polyline (-1,2,3 Formex) and a curvilinear abscissa "s". It returns the point on the polyline that has a curvilinear ditance s from the starting point. If s==0. it gives the starting point, if s==length of the polyline it returns the end point. For the end point a tollerance is used.
    NB s must be between 0 and total length of the polyline (with tolerance) otherwise it raises an error
    """
    if s==0:#questo serve per lo START-point della spezzata
        return P[0,0]

    max=0    #questo serve per l END-point della spezzata.
    for i in range(P[:].shape[0]): 
        max+=(((P[i,1]-P[i,0])**2).sum())**0.5 
        
    #if s==max:    #Ci metto una tolleranza
    if s>=max:
        if s<= max*(1.+rtol):return P[-1,1]
        if s<= max+atol:return P[-1,1]

    if s!=0 and s!=max:
        sup=0
        line=-1  
        rest=0    
        while sup<s:
            if line<P[:].shape[0]-1:
                line+=1
                inf=sup
                rest=s-inf
                sup+=(((P[line,1]-P[line,0])**2).sum())**0.5
            else:
                    raise ValueError,"curvilinear abscissa cannot be longer than the length of the polyline: %f." %max
        if line==-1:
                raise ValueError,"curvilinear abscissa must be positive or zero."
        len=(((P[line,1]-P[line,0])**2).sum())**0.5 #lunghezza del segmento    
        if rest!=0 and rest !=len:        
            r= P[line].reshape(1,2,3)            
            e= Formex(r).divide([0,rest/len,1])[:].reshape(2,2,3)
            new=zeros([P[:].shape[0]+1,2,3])
            new[:line]=P[:line]
            new[line+2:]=P[line+1:]
            new[line:line+2]=e
            return Formex(new)[line,1]
        if rest==0 or rest==len:
            return P[line,1]
            
def reDiscretizePolyline(PL, interv):
    """it takesthe Polyline PL and splits it at curvilineat abscissas contained in interv"""
    print "REDISC"
    print PL,interv
    newp=array([])
    for i in range(interv.shape[0]):
        newp=append (newp, SelectPoint(PL, interv[i]))
    newp = newp.reshape(-1, 3)
    print newp
    return newp
    
    


    
def BezierCurve( p0,t0, t1,p1 , npt=101, elongated=array([0., 0.])):
    """build a cubic spline between p0 and p1 that is tangent to p0t0 in p0 and p1t1 in p1. NB la distanza p0t0 determina la lunghezza del vettore tangente e lo stesso vale nel punto 1
    elongated: to extend the curve farther than the 2 vertices: gives 2 numbers that represents the fraction of curves elongated on the sides p0 and p1 respectively BUT this fraction is in terms of u!!! so ut is not linear with the curve length."""
    #u=array(range(npt))*1./(npt-1.)

    extmin= elongated[0]*npt
    extmax=(1.+elongated[1])*npt
    u=array(range(-extmin, extmax))*1./(npt-1.)#check the integers, it gives a Warning but it is not important!!!
    U=asmatrix([u**3., u**2., u, u*0.+1.])
    M=matrix([[-1., 3., -3., 1.], [3., -6., 3., 0.], [-3., 3., 0., 0.], [1., 0., 0., 0.]])#pag.440 of open GL
    P=asmatrix([p0, t0, t1, p1])
    c_p=array([])
    for i in range(U.shape[1]):
        c_p=append(c_p, U[:, i].reshape(4)*M*P)
    c_p=c_p.reshape(-1, 3)
    #uniforming the distance between the 2 consecutive points along the cubic line
    pl_c, len_c=polyline(c_p)    
    #curv_c=array(range(npt))*len_c/(npt*1.00001-1.)#if error occurres because of some rounding problems (rounding is not good curv_c[-1] can be higher than len_c) use :  curv_c=array(range(npt))*len_c/(npt*1.00001-1.)
    curv_c=array(range(npt))*len_c/(npt-1.)
    c_pu=reDiscretizePolyline(pl_c, curv_c)
    return c_pu
    
def CardinalSpline( p0,P0, P1,p1 , npt=101, elongated=array([0., 0.]), t=0.):
    """build a cardinal spline between P0 and P1. 
    elongated: to extend the curve farther than the 2 vertices: gives 2 numbers that represents the fraction of curves elongated on the sides P0 and P1 respectively. t=0.0 by default: Catmull-Rom or Overhauser splines
    NB the only difference between the CardinalSplines is the matrix M. So maybe they can be condensed into the same function"""
    #u=array(range(npt))*1./(npt-1.)

    extmin= elongated[0]*npt
    extmax=(1.+elongated[1])*npt
    u=array(range(-extmin, extmax))*1./(npt-1.)#check the integers!!!
    U=asmatrix([u**3., u**2., u, u*0.+1.])
    
    s=(1-t)/2.
    M=matrix([[-s, 2-s, s-2., s], [2*s, s-3., 3.-2*s, -s], [-s, 0., s, 0.], [0., 1., 0., 0.]])#pag.429 of open GL
    P=asmatrix([p0, P0, P1, p1])
    c_p=array([])
    for i in range(U.shape[1]):
        c_p=append(c_p, U[:, i].reshape(4)*M*P)
    c_p=c_p.reshape(-1, 3)
    #uniforming the distance between the 2 consecutive points along the cubic line
    pl_c, len_c=polyline(c_p)    
    curv_c=array(range(npt))*len_c/(npt-1.)#if error occurres because of some rounding problems (rounding is not good curv_c[-1] can be higher than len_c) use :  curv_c=array(range(npt))*len_c/(npt*1.00001-1.)
    c_pu=reDiscretizePolyline(pl_c, curv_c)
    return c_pu

def cardinalSplines(nod, npt=101, elongated=array([0., 0.])):
    pv=array([])
    for i in range(nod.shape[0]-3):
        if i==0:pt=CardinalSpline( nod[i], nod[i+1],  nod[i+2],  nod[i+3],  npt=npt, elongated=array([elongated[0], 0.]))
        if i==nod.shape[0]-4:pt=CardinalSpline( nod[i], nod[i+1],  nod[i+2],  nod[i+3],  npt=npt, elongated=array([0., elongated[1]]))
        if i!=0:
            if i!=nod.shape[0]-4: pt=CardinalSpline( nod[i], nod[i+1],  nod[i+2],  nod[i+3],  npt=npt, elongated=array([0., 0.]))
        pv=append(pv, pt)
    return pv.reshape(-1, 3)

        
# deprecated functions


def polyline(nod):
    print "! Deprecated: polyline; use the PolyLine class"
    E = PolyLine(nod)
    return E.asFormex(),E.length()


def naturalSpline(pt, endcond='notaknot', npts=101, elongation=[0., 0.]):
    NS = NaturalSpline(pt,endcond=[endcond,endcond],closed=endcond=='closed')
    return NS.points(npts-1,elongation)



clear()

print "An open natural spline"
P = Coords([[-1., 1., -4.], [1., 1., 2.],[2.6, 2., -4.], [2.9,  3.5, 4.], [2., 4., -1.],[1.,3., 1.], [0., 0., 0.], [0., -3., 0.], [2., -1.5, -2.], [1.5, -1.5, 2.], [0., -8., 0.], [-1., -8., -1.], [3., -3., 1.]])
draw(P, color='brown',marksize=6)
S = NaturalSpline(P, endcond='notaknot')
X = S.points(ndiv=20, extend=[0., 0.])
draw(X, color='green',marksize=2)
draw(PolyLine(X), color='turquoise', linewidth=7)
view('front')
zoomAll()


##Natural spline..Closed
P = Coords([[-5., -10., -4.], [-3., -5., 2.],[-4., 0., -4.], [-4.,  5, 4.], [6., 3., -1.], [6., -9., -1.]])
draw(P, color='yellow',marksize=8)
S = naturalSpline(P, endcond='closed', npts=21, elongation=[0.0, 0.0])
draw(S, color='green',marksize=2)
draw(PolyLine(S), color='green', linewidth=3)
view('front')
zoomAll()
exit()

##Cardinal spline
PP=array([[-1., 7., -14.], [-4., 7., -8.],[-7., 5., -14.],[-8., 2., -14.],  [-7.,  0, -6.], [-5., -3., -11.], [-7., -4., -11.]])
draw(Formex(PP.reshape(-1, 1, 3)), marksize=9, color='gray')
p=cardinalSplines( PP,  npt=31, elongated=array([0.0, 0.0]))
draw(Formex(p), marksize=2, color='gray')
#drawNumbers(Formex(p))
draw(polyline(p)[0], color='red', linewidth=4)

zoomAll()
#pause()

PP=array([[6., 7., 12.],[9., 5., 6.],[11., -2., 6.],  [9.,  -4., 14.]])
draw(Formex(array([PP[0], PP[1], PP[2], PP[3], ]).reshape(1, 4, 3)), color='white', linewidth=3)
#drawNumbers(Formex(array([PP[0], PP[1], PP[2], PP[3]]).reshape(-1, 1, 3)))
Pb=BezierCurve(PP[0], PP[1], PP[2], PP[3], npt=31, elongated=array([0., 0.]))
draw(Formex(Pb), marksize=2)

pPb, sp=polyline(Pb)
draw(pPb, linewidth=6, color='turquoise')

view('front')
zoomAll()
