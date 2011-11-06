#!/usr/bin/env awk
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

#
# Extract the nodal coordinates from a Gambit neutral file
#

BEGIN { 
  mode = 0;
  flag = 0;
  if (nodes == "") nodes="nodes.out"; 
  if (elems == "") elems="elems.out"; 
}

/^[ ]*NODAL COORDINATES/ { mode = 1; next; }
/^[ ]*ELEMENTS/ { mode = 2; next;}
/^[ ]*ENDOF/ { mode = 0; next; }


{ if (mode == 1) print $2,$3,$4 >> nodes;
  else if (mode == 2) {
	if (flag == 0) {print $4,$5,$6,$7,$8,$9,$10 >> elems; flag = 1;}
	else {print $1 >> elems; flag = 0;}	
}
  else next;
}
