.. $Id$    -*- rst -*-

..
  This file is part of the pyFormex project.
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  https://savannah.nongnu.org/projects/pyformex/
  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
  Distributed under the GNU General Public License version 3 or later.


  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see http://www.gnu.org/licenses/.



.. include:: <isonum.txt>
.. include:: defines.inc
.. include:: links.inc

.. _`WebGL`: http://en.wikipedia.org/wiki/WebGL
.. _`X Toolkit`: http://www.goxtk.com
.. _`WebGL BridgeDeck`: _static/BridgeDeck.html

WebGL models
============

`WebGL`_ is a fascinating emerging technology, available in most modern browsers (Chrome, Firefox, Opera, Safari).

With WebGL you can look at 3D models and manipulate them, almost just like
you would do inside pyFormex itself. pyFormex can now export most of its models
directly to a WebGL model inside an HTML page, so you can make your models
remotely available over the Web. These models use a slightly modified version
of the `X Toolkit`_.

If you have a WebGL enabled browser (for IE you will have to install third party plugins), you can have
a look at and play with some of such models through the links below.
Moving the mouse with the left button pressed will rotate the model. With the
middle mouse button pressed you can move the model. Pressing the right button
lets you zoom in and out. There is also a gui where you can interactively change
the color and transparency of the parts of the model, and even switch the parts
on or off.

- A bridgedeck model, by Frederik Anseeuw and Kenzo De Sutter.
  Try out the `WebGL BridgeDeck`_ model.

  .. image:: images/BridgeDeck.png
     :width: 60%
     :align: center
     :target: `WebGL BridgeDeck`_

.. End
