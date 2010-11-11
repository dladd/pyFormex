.. $Id$  -*- rst -*-
.. pyformex reference manual --- curve
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-curve:

:mod:`curve` --- Definition of curves in pyFormex.
==================================================

.. automodule:: curve
   :synopsis: Definition of curves in pyFormex.
   :members: circle,vectorPairAngle,convertFormexToCurve,distanceBetweenBezier

   ``Classes defined in module curve``


   .. autoclass:: Curve
      :members: pointsOn,pointsOff,ncoords,npoints,sub_points,sub_points_2,sub_tangents,lengths,pointsAt,subPoints,length,approx,toFormex

   .. autoclass:: PolyLine
      :members: nelems,toFormex,toMesh,sub_points,sub_points2,vectors,directions,avgDirections,lengths,atLength,reverse,split,cutWithPlane

   .. autoclass:: Polygon
      :members: 

   .. autoclass:: BezierSpline
      :members: pointsOn,pointsOff,part,sub_points,sub_tangents,length_intgrnd,lengths,approx_by_subdivision,reverse

   .. autoclass:: QuadBezierSpline
      :members: 

   .. autoclass:: CardinalSpline
      :members: 

   .. autoclass:: CardinalSpline2
      :members: sub_points

   .. autoclass:: NaturalSpline
      :members: compute_coefficients,sub_points

   .. autoclass:: Arc3
      :members: sub_points

   .. autoclass:: Arc
      :members: sub_points

   .. autoclass:: Spiral
      :members: 

   ``Functions defined in module curve`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

