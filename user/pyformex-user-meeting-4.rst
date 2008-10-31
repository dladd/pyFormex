.. This may look like plain text, but really is -*- rst -*-

====================================
pyFormex User Meeting 4 (2008-09-12)
====================================

Place and Date
--------------
These are the minutes of the pyFormex User Meeting of Friday September 12, 2008, at the Institute Biomedical Technology (IBiTech) of Ghent University, Belgium.

Participants
------------
The following pyFormex developers, users and enthusiants were present.

- Benedict Verhegghe, chairman,
- Matthieu De Beule, secretary,
- Peter Mortier,
- Sofie Van Cauter,
- Gianluca De Santis,
- Michele Conti
- Victor Ros.

Apologies
---------
- Wesley


Minutes of the previous meeting
-------------------------------
- No remarks


Agenda and discussion
---------------------
* pyFormex Developer Forum:

  - Benedict demonstrates the newly implemented fabulous *superspherical* transformation which can be considered an extension of the spherical coordinate transformation. The demonstration (which is in the SuperShape example) uses an input dialog that stays on the screen, lets the user fill in the parameters of a transformation, show the generated structure and even save the parameters to a file for later replay.

  - Running pyFormex from a BuMPix Live Linux USB key: the demo is postponed to the next meeting.

  - All example scripts have been given some keywords. These allow to create a catalog of examples according to keywords. 
  - Some new Input Dialog features have been implemented:

    + The input dialogs can now be non-modal (i.e. they stay open while the user continues to work with pyFormex). See SuperShape example.
    + Contents of the input widget can be changed programmatically from your scripts.
    + Work has started on some fancy input widgets like Sliders (see Lighting example)

  - Sofie demonstrates the possibilities to create tables with one or multiple tabs. Using more extended Qt4-functions, it is possible to create even more advanced dialogs. It could be interesting to add a table editor.



* pyFormex User Forum:

  - Gianluca uses the facility to import .wrl files from Philips Medical Systems into pyFormex. He presents his latest work on the automated creation of hexahedral meshes for FEA and CFD analysis based on the 3D angio data. Currently the method works well for non-bifurcated regions. Work on bifurcations is progressing.

  - Victor demonstrates the use of the *new* Bezier curves implementation for creating a parametric model of the geometry of a Hernia Patch.


* Discussion: Should pyFormex become a 'managed' project?

  In order to keep pyFormex a structured working environment and not disintegrate into a collection of separate scripts, it seems highly recommended to further development of pyFormex as a *managed* project. Therefore all attendees agree on this issue.
  Benedict will act as general manager and when necessary delegate subtasks to others:

  - Benedict will develop templates for examples, to create new scripts, for plugins, ...
  - Sofie will focus on surface and STL related stuff.
  - Matthieu volunteered to update the manual and tutorial. The tutorial should comprise simple things (e.g. exercises) and basic python/numpy (e.g. for loop, list, dictionary). The manual should comprise Tips&Tricks, and what to do when things go wrong.
  - Michele will focus on pre- and postprocessing for Finite Element simulations (FEAPpv).
  - Peter will continue to work on meshing algorithms.
  - Gianluca volunteers enthousiastically to work further on the creation of geometries from medical image data.



pyFormex challenge
------------------
Create a pyFormex model as closely as possible representing the famous `Klokke Roeland`_ bell in Ghent. The script should use the *superspherical* transformation. Solution (in the form of a pyFormex script) should be submitted by email to the `pyFormex project manager`_.

Varia
-----
- Some users have observed crashes of pyFormex on changing the rendering mode. After a bug was filed, the issue has been resolved (it involved an upgraded of the underlying OpenGL libraries). As a general rule, please report any problems and issue on the `pyFormex bug tracking`_ system, giving as much details as possible: what is the problem, when does it occur, repeatibility, installed versions of the software (paste the output of the Help->Detected Software menu item).
- The manual should include some information on how to run pyFormex from the `BuMPix`_ Live CD: how to use the configuration menu (e.g. change keyboard settings).
- pyFormex seems a very useful pre- and postprocessor for FEAPpv. If possible it could be interesting to include FEAPpv to the `BuMPix`_ Live CD/USB. Paraview is mentioned to be also a good postprocessing tool.
- It seems a good idea to summarize the most important pyFormex functionalities in a peer-reviewed manuscript.
- Interactive rectangle zooming is a highly needed tool for the pyFormex GUI.


Date of the next meeting
------------------------
The next meeting will be held at IBiTech on Friday, Oct 31, 2008 at 10.00h.


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
.. _`Klokke Roeland`: http://www.portalviajar.com/europa/belgica/gante/gante%2044%20-%20klokke%20roeland.JPG

.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

