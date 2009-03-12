.. This may look like plain text, but really is -*- rst -*-

====================================
pyFormex User Meeting 6 (2009-01-29)
====================================


Place and Date
--------------
These are the minutes of the pyFormex User Meeting of Thursday January 29, 2009, at the Institute Biomedical Technology (IBiTech), Ghent University, Belgium.


Participants
------------
The following pyFormex developers, users and enthusiasts were present.

- Benedict Verhegghe, chairman
- Matthieu De Beule
- Peter Mortier
- Sofie Van Cauter
- Gianluca De Santis, secretary
- Tomas Praet


Apologies
---------
- None


Minutes of the previous meeting
-------------------------------
- No remarks. The minutes are available from the `pyFormex User Meeting page`_.


Agenda and discussion
---------------------
There are three items on the agenda: the renewed draw command in the latest pyFormex version (0.7.4), the curve plugin and the Laplace smoothing algorythm for STL surfaces.

* Draw command:

  - Benedict showed the capabilities of the draw command. With respect to the prevoius version, more parameters have been set automatically in a "clever" way in order to give the most useful representation of the Formices in viewports. It has been shown how to use multiple viewports, how to handle the different options in the draw command whenever the default set of parameters would not be satisfying. With respect to the previous version of pyFormex, the new Formices are cumulated on the graphycs by default, unless the clear() command is used. It allows to cumulate objects from the same script or different scripts. Thus, the zoom can be either set automatically to the last object or can be fixed to a specific object (e.g. the biggest) so that other objects (e.g. smaller details) are added without changing the view.
  -a bug has been shown when the draw command is used a lot of time: the quality of te image decreases and sometimes parts of the image disappeare. It has been also shown that restarting pyFormex solves the problem. However, this bug will be fixed.

* Curve plugins:

  -Benedict has demonstrated a new capability of pyFormex to draw splines based on list of control points. A number of curves have been implemented by Gianluca and Benedict in pyFormex in the dedicated plugin: Bezier, Cardinal Splines, Natural Splines, PolyLines. The curves can be drawn changing some parameters (tension, end-condition, curliness, open/closed etc.). These curves are also available in commercial softwares as Rhinoceros but in pyFormex the user can have an higher control on the parameters needed for a specific application. For each curve it is possible to extract an arbitrary number of points using the length parameter t that appears in the system of equations which represent the analytical description of the curve. A concern regards the representation of a point that has a specific curvilinear distance from the starting of the curve. For the moment this problem is solved by approximating of the curve with an aribitrarily high number of line segments (PolyLine). In the future, an analytical solution will give the exact solution to the problem.

* An application of the curve plugins to reconstruct vessel geometries:

   -Gianluca has used the curve to reconstruct a vessel lumen starting from some points coming out of 3D angiographic imaging of coronary tree. Two sets of curves, one circumferential and the other longitudinal, have been used to reconstract non-symmetric tube-like surfaces. This approach allows to reconstruct tube-pieces that can be then joined together using Bezier surfaces by controlling the slope (G1 continuity) at the sides of the connecting surfaces.
   -After the reconstruction of the surfaces, the volume is filled using iso-parametric transformation that can fit a structured mesh (e.g. a meshed parallepiped) into an arbitrary volume using a parabolic/cubic interpolation. This feature of pyFormex relies on the capability to transform directly meshes, while normal package needs first to define a geometry and afterwards to mesh it. 

*  Smoothing algorythm for triangulated surfaces

   -Sophie has implemented a smoothing filter for triangulated surfaces (STL). This filter uses Laplacian filter to smooth the surface in an iterative way (e.g. it is possible to smooth keeping the 1 derivative continuity). The method calculate a new point basing on the coordinated of the closest points. Sophie has also compared her smoothing function to smoothing function of the GTS package and a positive feature has been verified: the Sophie's algorythm does not modifify the volume of the object whole the GTS fucntion visibly shrink the object. This feature is obtained by combining two filters at the same time. It is possible that Rhinoceros applies the same smoothing filter. Eventually, an elegant function to export data from pyFormex to Excel has been demonstrated.

   -Tomas has appplied the smoothing algorythm of Sophie to the Ankle-Foot Orthersis (AFO) not only to smooth the surface but also to increase the quality of STL surfaces. He was able to quantify the quality of a triangulate mesh using the same criteria descripted in the Abaqus documentation. These quality parameters have been visulized graphically as colors on the mesh and clearly showed the improvement of the mesh after applying the smoothing filter.

*  Other points

   -Other concerns during the meeting were:
	+   running pyFormex on multiple CPUs. The preferred solution is to avoid an ad hoc implementation but rather to use the new parallel numerical python
	+   Hexahedral mesh in pyFormex. Hex elements in pyFormex are defined with the same convention of Abaqus. Although the user can difine a correct original topology of a mesh, a number of pyFormex transformation (e.g. mirror, reflect, connect, sweep, revolve) can return a chiral copy of the hexahedrals. The chiral copy is not conceptually correct  and needs to be corrected as soon as possible before continuing to elaborate the mesh. A function to convert a mesh with correct and uncorrect hexahedrals into a mesh with all correct hexahedrals has been developed by Gianluca.


Varia
-----
  - A previous contest has been renewed: a script to generated the shape of the Roeland bell. It will be an easy task thank to the the curve plugin.


Date of the next meeting
------------------------
The next meeting will be held at IBiTech in March 2009.


.. Here are the targets referenced in the text

.. _`pyFormex website`: http://pyformex.berlios.de/
.. _`pyFormex home page`: http://pyformex.berlios.de/
.. _`pyFormex user meeting page`: http://pyformex.berlios.de/usermeeting.html
.. _`pyFormex developer site`: http://developer.berlios.de/projects/pyformex/
.. _`pyFormex forums`: http://developer.berlios.de/forum/?group_id=2717
.. _`pyFormex developer forum`: https://developer.berlios.de/forum/forum.php?forum_id=8349
.. _`pyFormex bug tracking`: http://developer.berlios.de/bugs/?group_id=2717
.. _`pyFormex project manager`: mailto:benedict.verhegghe@ugent.be
.. _`UGent digital learning`: https://minerva.ugent.be/main/ssl/login_en.php
.. _`pyFormex news`: http://developer.berlios.de/news/?group_id=2717
.. _`pyformex-announce`: http://developer.berlios.de/mail/?group_id=2717
.. _`IBiTech`: http://www.ibitech.ugent.be/
.. _`BuMPix`: ftp://bumps.ugent.be/pub/bumpix/
.. _`Debian Live Project`: http://wiki.debian.org/DebianLive/Howto/USB/
.. _`WinSCP`: http://winscp.net/eng/index.php

.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

