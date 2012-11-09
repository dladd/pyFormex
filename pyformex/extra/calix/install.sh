#!/bin/bash
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
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
PKG=calix
VERSION=1.5
RELEASE=$VERSION-a8
PKGVER=$PKG-$VERSION
NAME=$PKG-$RELEASE
ARCHIVE=$NAME.tar.gz
DIR=$PKGVER
URI=ftp://bumps.ugent.be/pub/calix/$ARCHIVE

_usage() {
    cat <<EOF
This script helps with installing calix from source:

Prefered installation (in /usr/local):

./install.sh get unpack make
sudo ./calpy_install install
./calpy_install clean

Use at your own risk if you do not understand what is happening!
EOF
}

_get() {
    [ -f $ARCHIVE ] || wget $URI
}

_unpack() {
    rm -rf $DIR
    tar xvzf $ARCHIVE
}

_make() {
    pushd $DIR
    make
    popd
}

_install() {
    [ "$EUID" == "0" ] || {
	echo "install should be done as root!"
	return
    }
    pushd $DIR
    make install
    popd
}


_clean() {
    rm -rf $DIR
    rm -f $ARCHIVE
}

[ -z "$@" ] && { set usage; }

for cmd in "$@"; do

    case $cmd in 
	get | unpack | patch | make | install | rename | clean ) _$cmd;;
	all ) _get;_unpack;_make;_install;_clean;;
        * ) _usage;;
    esac

done
