.. This may look like plain text, but really is -*- rst -*-

====================================
pyFormex User Meeting 1 (2008-03-11)
====================================

Place and Date
--------------
These are the minutes of the second pyFormex User Meeting, which took place on Tuesday March 11, 2008, at the Institute Biomedical Technology (IBiTech) of Ghent University, Belgium.

Participants
------------
The following pyFormex developers, users and enthusiants were present.

- Benedict Verhegghe, project manager and chairman of the meeting,
- Matthieu De Beule,
- Peter Mortier,
- Sofie Van Cauter,
- Gianluca De Santis,
- Jan Belis,
- Wesley Vanlaere,
- Pieter Vanderhaeghe,
- Wouter Vanrenterghem.

Apologies
---------
Michele Conti.

Minutes of the previous meeting
-------------------------------
No remarks. The minutes are available on the `pyformex website`_.

Agenda and discussion
---------------------
* pyFormex Developer Forum:
  
  - The `pyFormex project manager`_ presents some recent developments and ideas for future work:

    - Combining different types of grids is now possible and currently available in the *fe* plugin (seperate node lists are merged into one list).
    - The current procedure to assign properties will be modified in order to allow assignment of a property to a list of object elements instead of via the object property numbers.
    - The difference between transforming a Formex or a Surface model is explained. The original coordinates remain unchanged when transforming a Formex model, whereas they are lost when transforming a Surface model. This is because the size of a Surface models is usually larger. This may change in future.
    - The intersection of a surface with a plane has been modified as it resulted in an error in some cases (e.g. triangle laying completely in the intersecting plane).
    - The use of a project is illustrated. All the global variables are saved from the moment you start a project until it is closed. The button showing the project name in the left bottom corner should disappear after closing the project.
    - Postprocessing is currently mainly limited to beam elements. This will be extended to other element types.
    - Picking causes a change in height of the canvas, because the height of the buttons which appear during the picking operation is larger than the height of the bar in which they appear.

* pyFormex User Forum:

  - Pieter reported that creating an Abaqus input file with only one element group gives an error. This will be fixed.
  - Peter proposed to create a general FE menu to create FE input files using the GUI. A geometry could be imported during a first step. Subsequent steps could involve the assignment of sections, materials, boundary conditions and loads. Matthieu suggested to incorporate export possibilies to different FE solvers. Benedict added that all functionalities should be accessible through scripting. 
  - Matthieu gave an overview of the remarks of the civil engineering students who used pyFormex during a project (some of these suggestions have already been implented):

    - General:

      - Visualize coordinate system
      - Include an option to query coordinates, distances, etc.
      - Information should be provided on how to connect to the internet from the Linux Live CD
   
    - Manual:

      - It could be useful to link each definition to an example demonstrating the use of the definition
      - Extend keywords list

    - Introduction:

      - Include a short introduction to Python
      - Extend the introduction to pyFormex with more demo's, explain the structure of a Formex, etc.
      - Include a demo which illustrates the translation of an idea into a script

    - Preprocessing:

      - Create loading and boundary conditions interactively
      - A template for exporting to FE solvers should be provided

    - Postprocessing:

      - Set scale interactively
      - Request specific output in specific element (e.g. N in beam element)

  - Gianluca has been working on the interface between pyFormex and Fluent. Fluent uses different conventions. Peter suggested to add the conventions used in pyFormex to the manual for different element types.
  - Jan proposed to add references of scientific publications making use of pyFormex to the website.

* pyFormex contest:

  - There was one splash screen submitted by Peter and this is added to the pyFormex distribution. Benedict also developed a tool so that users can select their favourite splash screen. Users can also see a preview of the splash screen.


Date of the next meeting
------------------------
The next meeting will be held Tuesday, April 8, 2008 at 10.00h.


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

