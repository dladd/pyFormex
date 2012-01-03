.. $Id$   -*- rst -*-

Overzicht van de pyFormex software
==================================

Wat is pyFormex
---------------
pyFormex is een open source software voor de creatie, manipulatie en visualisatie van grote geometrische modellen in 3D.

pyFormex wordt momenteel ontwikkeld onder leiding van prof. Benedict Verhegghe van de Universiteit Gent en wordt verspreid onder de GNU General Public License v2.0 of hoger via de website http://pyformex.org.


Opvatting
---------
pyFormex is in de eerste plaats een verzameling van algoritmen en data-objecten, en een script-taal waarmee vlot allerhande toepassingen omtrent de behandeling van 3D geometrieën kunnen gemaakt en getest worden.

Applicaties die reeds verwezenlijkt werden liggen o.a. in volgende domeinen:

- genereren van parametrische geometrische modellen,
- creatie van eindige-elementen meshes,
- bewerken van 3D surfaces bekomen uit medical imaging (CT,MRI),

Typische pyFormex scripts bestaan meestal uit een opeenvolging van mathematische transformaties, tekeninstructies voor het afbeelden van de geometrie, en interactieve operaties.

pyFormex is door zijn opvatting gemakkelijk uitbreidbaar, ideaal voor parametrisch ont-werpen, en kan gemakkelijk aan externe software gekoppeld worden.

Overzicht van de componenten
----------------------------
De onderstaande figuur geeft een organigram van de voornaamste pyFormex componenten.

.. figure:: pyformex-arch.png
   :scale: 50  
   :alt: pyFormex components
   :target: pyformex componenten
   
   Schematisch overzicht van de voornaamste pyFormex componenten.
   De externe onderdelen zijn: Python, Numpy, OpenGL, PyOpenGL, Qt4, PyQt4.
 

Python_
.......
De Python programmeertaal vormt de universele lijm tussen de verschillende componenten van pyFormex. Het is meteen ook de implementatie van de pyFormex scripting taal.

NumPy_
......
NumPy (Numerical Python) is een performante implementatie van numerieke matrices voor Python. Dit vormt de basis voor de mathematische bewerkingen in pyFormex.

OpenGL_
.......
OpenGL is de industrie-standaard voor de ontwikkeling van cross-platform, interactieve, 2D and 3D grafische applicaties. Wordt in pyFormex gebruikt voor het tekenen en interactief manipuleren van 3D geometrie.

PyOpenGL_
.........
De Python verbinding met OpenGL.

Qt4_
.....
Qt is een cross-platform ontwikkelomgeving voor interactieve applicaties. Hierop is de grafische gebruikersomgeving van pyFormex gebouwd.


PyQt4_
......
De Python verbinding met Q4.


Huidige mogelijkheden
---------------------

pyFormex is nog volop in ontwikkeling maar biedt al voldoende mogelijkheden voor subprojecten om op verder te bouwen. Het onderstaande is een lijstje van de voornaamste kenmerken aanwezig in de versie 0.6.

- Belangrijkste data-objecten: 
  
  - Coords, een verzameling punten in een 3D ruimte,
  - Formex, een gestructureerde verzameling punten waarin sets van punten de
    betekenis krijgen van een geometrisch object (lijn, driehoek, oppervlak, ...),
  - Surface, een specifieke implementatie van ruimtelijke oppervlakken.

- Belangrijkste interne functies:

  - In- en uitvoer van geometrie (o.a. lezen van STL type bestanden)
  - Creatie van geometrie (ab initio)
  - Transformatie van coordinaten van de geometrie (met behoud topologie) 
  - Transformatie met veranderende topologie (bv. snijden, extrusie,...)
  - Visualisatie van de geometrie (wireframe, flat en smooth rendering)
  - Een begin van interactieve bewerkingstools
  - Persistente opslag van projecten
  - Een krachtige scripting taal die toelaat alle bewerkingen te automatiseren
  - Open GUI architectuur die door de gebruiker kan gewijzigd worden
  - Een uitgebreide set voorbeelden van scripts
  - Omzetting van de geometrie naar eindige-elementen (EE) modellen
  - Enkele pre- en postprocessing functies voor EE-simulaties
  - Opslaan van beelden in verschillende image formaten 

- Bindingen met uitwendige software:

  - admesh: behandelen van STL oppervlakken
  - tetgen: volumevermazing binnen STL oppervlakken
  - gl2ps: opslaan van OpenGL rendering in (E)PS formaat
  - calpy: EE module voor lineaire elasticiteit
  - gts: library met functies voor operaties op STL oppervlakken
  - Abaqus: commercieel EE-pakket 

  Deze externe pakketten kunnen binnen pyFormex gebruikt worden of er mee
  samenwerken,
  maar zijn niet vereist voor de werking van pyFormex. Zij bieden natuurlijk
  wel extra functies indien aanwezig.



.. _pyformex: http://pyformex.org/
.. _python: http://www.python.org/
.. _numpy: http://numpy.scipy.org/
.. _opengl: http://www.opengl.org/
.. _pyopengl: http://pyopengl.sourceforge.net/
.. _qt4: http://trolltech.com/products/qt
.. _pyqt4: http://www.riverbankcomputing.co.uk/pyqt/

.. target-notes::

