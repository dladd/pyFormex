#!/bin/bash
# $Id$
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


PKG=sippy-ftgl
PKGVER=$PKG-0.1 
ARCHIVE=$PKGVER.tgz
URL=http://sippy-ftgl.googlecode.com/files/$ARCHIVE

[ -f $ARCHIVE ] || wget $URL
rm -rf $PKG
tar xvzf $ARCHIVE
pushd $PKGVER

# !! We should check for libftgl, libfgtl dev, freetype2 dev
# also python-sip-dev!
# This supposes they are all installed in the same base, either
# /usr/local (default) or /usr
[ -f "/usr/lib/libftgl.so.2" ] && {
    sed -i 's|/usr/local|/usr|g' configure.py
}

python configure.py
make
make install

popd
#rm -rf $PKG
