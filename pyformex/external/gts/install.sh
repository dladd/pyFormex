#!/bin/bash
# $Id: install.sh 1583 2010-10-11 16:49:51Z bverheg $
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
# This script helps with installing the gts library and utilities from source:
#
# Prerequisites: libglib2.0-dev
#
# ./gts_install get unpack patch make
# sudo ./gts_install install rename
# ./gts_install clean
#
# Use at your own risk if you do not understand what is happening!
#

VERSION=0.7.6
NAME=gts-$VERSION
ARCHIVE=$NAME.tar.gz
URI=http://prdownloads.sourceforge.net/gts/$ARCHIVE

examples="cartesian cleanup coarsen delaunay gtstoc iso merge oocs optimize partition refine set smooth sphere split stripe transform traverse volume"

_get() {
    [ -f $ARCHIVE ] || wget $URI
}

_unpack() {
    rm -rf $NAME
    tar xvzf $ARCHIVE
}

_make() {
    pushd $NAME
    ./configure
    make
    popd
}

_install() {
    [ "$EUID" == "0" ] || {
	echo "install should be done as root!"
	return
    }
    pushd $NAME
    make install
    ldconfig
    popd
}

_patch() {
    pushd $NAME
    patch -p0 < ../gts-0.7.6-bv.patch
    popd
}

# !! Only examples delaunay and transform are installed.
# Better copy from build directory
_rename() {
    [ "$EUID" == "0" ] || {
	echo "install should be done as root!"
	return
    }
    for name in $examples; do
	gtsname=gts${name#gts}
	src=/usr/local/bin/$name
	tgt=/usr/local/bin/$gtsname
	[ -f "$src" -a "$src" != "$tgt" ] && mv $src $tgt
    done
}

_clean() {
    rm -rf $NAME
    rm -f $ARCHIVE
}

for cmd in "$@"; do

    case $cmd in 
	get | unpack | patch | make | install | rename | clean ) _$cmd;;
	all ) _get;_unpack;_patch;_make;_install;_rename;_clean;;
        * ) echo "UNKNOWN command $cmd";;
    esac

done
