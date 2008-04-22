.. This may look like plain text, but really is -*- rst -*-

====================================
pyFormex User Meeting 1 (2008-04-08)
====================================

Place and Date
--------------
These are the minutes of the third pyFormex User Meeting, which took place on Tuesday April 8, 2008, at the Institute Biomedical Technology (IBiTech) of Ghent University, Belgium.

Participants
------------
The following pyFormex developers, users and enthusiants were present.

- Benedict Verhegghe, `pyFormex project manager`_ and chairman of the meeting,
- Peter Mortier,
- Sofie Van Cauter,
- Gianluca De Santis,
- Wesley Vanlaere,
- Pieter Vanderhaeghe,
- Wouter Vanrenterghem.

Apologies
---------
- Matthieu De Beule
- Michele Conti

Minutes of the previous meeting
-------------------------------
No remarks. The minutes are available on the `pyformex website`_.

Agenda and discussion
---------------------
* pyFormex Developer Forum:
  
  - Benedict presents some recent developments:

    - The postprocessing functionalities have been extended. It is now possible to render values on a mesh (e.g. displacements, stresses, etc.).
    - The impact of using a averaged normals on surface rendering is shown. The surfaces seem much smoother when using the averaged normals. The calculation of the averaged normals is currently too slow for use with very large structures, but a significant speed-up will be obtained from implementing it in C.

  - Sofie demonstrates some of the tools she has been working on:

    - A new method to pick elements has been implemented. In addition to the standard picking procedure (which picks all elements within a box), it is now also possible to only pick those elements within the box that are connected to elements in the previous selection.

    - Renaming variables can be very useful when dealing with projects because the variables are saved within the project. This is now possible in the tools menu.
    - Cutting a surface with one or multiple planes is now much easier to do. The user is able to define and visualize a plane (with the tools menu) which facilitates the cutting (surface menu). Planes can be defined by giving three points or by one point and a normal.


* pyFormex User Forum:

  - Pieter extended the interface with Abaqus(R). Rigid and membrane elements have been added, more advanced material definitions are now possible (plastic behaviour, damping, etc.), interaction and interaction properties can be defined and the user is now able to create input files for explicit simulations. Benedict will include these new functionalities in the pyFormex distribution.
  - Benedict will continue to work on the postprocessing of FEA/CFD results.
  - Wouter uses pyFormex to create a parametric finite element model of a cylindrical tank. He proposes to add the possibility to visualize boundary conditions and loads. For these simulations, the Abaqus interface should be extended with the option to do buckling analyses.
  - Gianluca shows some of his results regarding the modelling of restenosis. This includes a closed cycle of creating a model in pyFormex, running a CFD analysis using Fluent(R), read the results back into pyFormex, make changes to the geometry of the model, and start a new cycle, until convergence is obtained.


Date of the next meeting
------------------------
The next meeting will be held Tuesday, May 6, 2008 at 10.00h.


.. Here are the targets referenced in the text

.. _`pyFormex website`: http://pyformex.berlios.de/
.. _`pyFormex home page`: http://pyformex.berlios.de/
.. _`pyFormex developer site`: http://developer.berlios.de/projects/pyformex/
.. _`pyFormex forums`: http://developer.berlios.de/forum/?group_id=2717
.. _`pyFormex developer forum`: https://developer.berlios.de/forum/forum.php?forum_id=8349
.. _`pyFormex bug tracking`: http://developer.berlios.de/bugs/?group_id=2717
.. _`pyFormex project manager`: mailto:benedict.verhegghe@ugent.be
.. _`UGent digital learning`: https://minerva.ugent.be/main/ssl/login_en.php
.. _`pyFormex news`: http://developer.berlios.de/news/?group_id=2717
.. _`pyformex-announce`: http://developer.berlios.de/mail/?group_id=2717
.. _`IBiTech`: http://www.ibitech.ugent.be/

.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

