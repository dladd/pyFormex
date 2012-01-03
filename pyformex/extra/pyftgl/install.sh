#!/bin/bash
# $Id: install.sh 1583 2010-10-11 16:49:51Z bverheg $
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
# Install Python wrapper for ftgl
#
[ "$1" = "all" ] || exit

PYFTGL=PyFTGL-0.4b 
PYFTGL_TGZ=$PYFTGL.tar.gz
PYFTGL_URL=http://pyftgl.googlecode.com/files/$PYFTGL_TGZ

[ -f $PYFTGL_TGZ ] || wget $PYFTGL_URL
rm -rf $PYFTGL
tar xvzf $PYFTGL_TGZ
pushd $PYFTGL
python setup.py build
python setup.py install
popd
#rm -rf $PYFTGL
