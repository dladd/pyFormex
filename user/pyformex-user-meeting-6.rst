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
- The minutes of the prvious meeting were not yet available. They will be approved later and put on the `pyFormex User Meeting page`_.


Agenda and discussion
---------------------
There are three items on the agenda: the renewed draw command in the latest pyFormex version (0.7.4 alpha), the curve plugin and the Laplace smoothing algorythm for STL surfaces.

* Changes to the *draw* command:

  Benedict explains the changes to the draw command. Default values of the draw command arguments and their interaction with the DrawOption settings are now handled in a more uniform way. With respect to the previous version, more parameters have been set automatically in a "clever" way in order to give the most useful representation of the Formices in viewports. It was shown how to use multiple viewports, how to handle the different options in the draw command whenever the default set of parameters would not be satisfying. 

  The most important change is that *bbox=None* argument will no longer keep the focus unchanged, but instead will use the DrawOption setting, whatever its current value is. To keep the focus unchanged, use *bbox='last'*. Since this is the default DrawOption setting, most scripts will still run as expected as long as you have not changed the DrawOption settings. Updating your scripts is highly recommended to avoid unexpected results.

  
  A bug is reported. When the draw command is used a lot of time, the quality of te image decreases and sometimes parts of the image disappears. Restarting pyFormex solves the problem. This bug will be fixed.

* Curve plugins:

  Benedict demonstrates a new capability of pyFormex to draw splines based on lists of control points. A number of curves have been implemented by Gianluca and Benedict in pyFormex in the *curves* plugin: Bezier, Cardinal Splines, Natural Splines, PolyLines. The curves can be drawn changing some parameters (tension, end-condition, curliness, open/closed etc.). Compared to most commercial software implementing such curves, the pyFormex user has a higher control on the parameters needed for a specific application. For each curve it is possible to extract an arbitrary number of points using the length parameter t that appears in the system of equations which represent the analytical description of the curve. A concern regards the representation of a point that has a specific curvilinear distance from the starting of the curve. For the moment this problem is solved by approximating of the curve with an aribitrarily high number of line segments (PolyLine). In the future, an analytical solution may give the exact solution to the problem.

* An application of the curve plugins to reconstruct blood vessel geometries:

  Gianluca has used the curve to reconstruct a vessel lumen starting from some points resulting from 3D angiographic imaging of the coronary tree. Two sets of curves, one circumferential and the other longitudinal, have been used to reconstruct non-symmetric tube-like surfaces. This approach allows to reconstruct tube-pieces that can be then joined together using Bezier surfaces by controlling the slope (G1 continuity) at the sides of the connecting surfaces.
  
  After the reconstruction of the surfaces, the volume is filled using iso-parametric transformation mapping a structured mesh (e.g. a meshed parallepiped) onto an arbitrary volume using a parabolic/cubic interpolation. This feature of pyFormex relies on the capability to directly transform meshes, while most other Finite Element modelling packages first define a geometry and mesh it afterwards. 

* Smoothing algorithm for triangulated surfaces

  Sofie has implemented a smoothing filter for triangulated surfaces (STL). This filter uses a Laplacian filter to smooth the surface in an iterative way (e.g. it is possible to smooth with first derivative continuity). The method calculates a new point based on the coordinates of the closest points. Sofie has also compared her smoothing function to the smoothing function of the GTS package and a positive feature has been verified: Sofie's algorithm does not modify the volume of the object while the GTS function visibly shrinks the object. This feature is obtained by combining two filters at the same time.

  Tomas has appplied the smoothing algorithm of Sofie to an Ankle-Foot Orthosis (AFO) not only to smooth the surface but also to increase the quality of STL surfaces for use in FEA. He was able to quantify the quality of a triangulate mesh using the same criteria descripted in the Abaqus documentation. These quality parameters have been visualized graphically as colors on the mesh and clearly showed the improvement of the mesh after applying the smoothing filter.

* Running pyFormex on multiple CPUs.

  The preferred solution is to avoid an ad hoc implementation but rather to use the parallel Numerical Python (numpy) infrastructure that is currently under development. 
	
* Hexahedral mesh in pyFormex. 
  
  Hex elements in pyFormex are defined with the same convention of Abaqus. Although the user can define a correct original topology of a mesh, a number of pyFormex transformation (e.g. mirror, reflect, connect, sweep, revolve) can return a chiral copy of the hexahedrals. The chiral copy is not conceptually correct  and needs to be corrected as soon as possible before continuing to elaborate the mesh. A function to convert a mesh with correct and uncorrect hexahedrals into a mesh with all correct hexahedrals has been developed by Gianluca.


Varia
-----
  - Sofie demonstrates an elegant function to export data from pyFormex to Excel.

  - A previous contest has been renewed: a script to generated the shape of the Roeland bell. It should be an easy task thanks to the the curve plugin.


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

