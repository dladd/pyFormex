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
  
Screenshots
-----------

.. _`Wire stent`: _images/.png
.. _`Surface mesh manipulations`: _images/intersection.png
.. _`Fluid simulations`: _images/cfd.png
.. _`Slicing operation`: _images/HorseSlice.png

- Stent unroll procedure: This screenshot shows four stages in the
  evaluation of the geometry of a stent device.

.. image:: images/stent_unroll.png
   :align: center

- Transparency: The transparent mode can be useful for many
  applications, such as the one shown in this screenshot. By using the
  transparent mode, it is possible to visualize the complete stent
  structure, which is implanted in a diseased artery.

.. image:: images/transparency.png
   :align: center

- `Wire stent`_: This screenshot shows the main stages during the
  generation of the wire stent geometry.

.. image:: images/WS_screenshot.png
   :align: center

- `Surface mesh manipulations`_: The top viewports show a stent device
  (acquired from CT scan images) in smooth rendering. At the bottom,
  some cross section are shown which have been created with pyFormex.

.. image:: images//intersection.png
   :align: center

- `Fluid simulations`_: pyFormex can also be used to create grids for
  Computational Fluid Dynamics (CFD). Again, many parameters can
  easily be modified (angle of intersection, number of boundary
  layers,...).

.. image:: images/cfd.png
   :align: center

- `Slicing operation`_: In this example pyFormex was used to slice a
  surface model of a Horse. From left to right and top to bottom: the
  original surface, 40 slices in the x-direction, 50 slices in the
  direction of the (x,y)-bisector, and 100 slices in the
  z-direction. Operations like these could e.g. be used to create
  models for rapid prototyping..

.. image:: images//HorseSlice.png
   :align: center

And here we have some images of pyFormex running on ARM-based hardware:

- `efika_smartbook`_: A Genesi Efika MX Smartbook running pyFormex.
- `efika_screenshot`_: Screenshot of the Genesi Efika running pyFormex
  under Ubuntu 10.10.

And finally, a first glimpse of

- `imac_screenshot`_: pyFormex running on an iMac.

.. _`efika_smartbook`: _static/efika_smartbook.jpg
.. _`efika_screenshot`: _static/efika_screenshot.png
.. _`imac_screenshot`: _static/pyformex_on_Mac.png

.. End
