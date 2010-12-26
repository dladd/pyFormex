DEBUG: Using the (slower) Python misc functions
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
   :members: circle,convertFormexToCurve

   ``Classes defined in module curve``


   .. autoclass:: Arc
      :members: sub_points_2,sub_directions,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write

   .. autoclass:: Arc3
      :members: sub_points_2,sub_directions,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write

   .. autoclass:: BezierSpline
      :members: sub_points_2,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write,pointsOn,pointsOff,part,sub_points,sub_directions,sub_tangents,length_intgrnd,lengths,approx_by_subdivision,extend,reverse

   .. autoclass:: CardinalSpline
      :members: sub_points_2,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write,pointsOn,pointsOff,part,sub_points,sub_directions,sub_tangents,length_intgrnd,lengths,approx_by_subdivision,extend,reverse

   .. autoclass:: CardinalSpline2
      :members: sub_points_2,sub_directions,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write

   .. autoclass:: Curve
      :members: sub_points,sub_points_2,sub_directions,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write

   .. autoclass:: NaturalSpline
      :members: sub_points_2,sub_directions,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write

   .. autoclass:: PolyLine
      :members: sub_points_2,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,toMesh,write,sub_points,sub_points2,sub_directions,vectors,directions,avgDirections,lengths,atLength,reverse,split,cutWithPlane

   .. autoclass:: Polygon
      :members: sub_points_2,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,toMesh,write,sub_points,sub_points2,sub_directions,vectors,directions,avgDirections,lengths,atLength,reverse,split,cutWithPlane

   .. autoclass:: QuadBezierSpline
      :members: sub_points_2,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write,pointsOn,pointsOff,part,sub_points,sub_directions,sub_tangents,length_intgrnd,lengths,approx_by_subdivision,extend,reverse

   .. autoclass:: Spiral
      :members: sub_points,sub_points_2,sub_directions,sub_directions_2,pointsAt,directionsAt,affine,bump,bump1,bump2,cylindrical,egg,flare,isopar,map,map1,mapd,projectOnCylinder,projectOnSphere,reflect,replace,rollAxes,rot,rotate,scale,shear,spherical,superSpherical,swapAxes,toCylindrical,toSpherical,transformCS,translate,trl,subPoints,length,copy,approx,toFormex,write

   ``Functions defined in module curve`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

