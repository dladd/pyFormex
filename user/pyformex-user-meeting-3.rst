.. This may look like plain text, but really is -*- rst -*-

====================================
pyFormex User Meeting 3 (2008-05-06)
====================================

Place and Date
--------------
These are the minutes of the fourth pyFormex User Meeting, which took place on Tuesday May 6, 2008, at the Institute Biomedical Technology (IBiTech) of Ghent University, Belgium.

Participants
------------
The following pyFormex developers, users and enthusiants were present.

- Benedict Verhegghe, `pyFormex project manager`_ and chairman of the meeting,
- Matthieu De Beule,
- Peter Mortier,
- Sofie Van Cauter,
- Gianluca De Santis.

Apologies
---------
- Michele Conti,
- Wesley Vanlaere,
- Pieter Vanderhaeghe,
- Wouter Vanrenterghem.


Minutes of the previous meeting
-------------------------------
No remarks. The minutes are available on the `pyformex website`_.

Agenda and discussion
---------------------
* pyFormex Developer Forum:
  
  - Benedict presents some recent developments:

    - The property system is now more or less definitive. A PropertyDB Class (pyformex/plugins/properties.py) has been defined, which collects all properties that can be set on a geometrical model. Some examples are materials, sections, node properties and element properties. More information can be found in the pyFormex tutorial (2.4 Assigning properties to geometry) and in the examples file (Analysis/FeAbq.py, Analysis/FeEx.py). Playing the latter script will generate a menu, which can be used to interactively create a simple finite element model and run an analysis.
    - Some tools from the jobs menu (pyformex/local) are illustrated, namely how to submit an Abaqus job to the cluster and check the results and get them back from the cluster.
    - The postabq converser, which translates an Abaqus .fil file into a postprocessing script, will now generate a .post file instead of a -post.py file.
    - A fast C version for showing averaged normals on surface rendering has been implemented.
    - The use of filters (single,closest,connected) during picking operations is demonstrated.


* pyFormex User Forum:

  - Some suggestions were made for future work:

    - reading VRML files
    - creating a native STL reader
    - working with nurbs surfaces


Date of the next meeting
------------------------
The next meeting will be held Thursday, July 3, 2008 at 10.00h.


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

