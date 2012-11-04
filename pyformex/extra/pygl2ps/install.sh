#!/bin/bash
# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
# Install Python wrapper for gl2ps
# If you change version, you need to adapt this file, gl2ps.i, setup.py
# and remove the gl2ps_wrap.c and gl2ps.py
#
[ "$1" = "all" ] || exit

GL2PS=gl2ps-1.3.3
GL2PS_TGZ=$GL2PS.tgz
GL2PS_URL=http://geuz.org/gl2ps/src/$GL2PS_TGZ


[ -f $GL2PS_TGZ ] || wget $GL2PS_URL
[ -f gl2ps_wrap.c ] || swig -python gl2ps.i
rm -rf $GL2PS
tar xvzf $GL2PS_TGZ
cp *.c *.py $GL2PS
pushd $GL2PS
python setup.py build
python setup.py install
popd
rm -rf $GL2PS
