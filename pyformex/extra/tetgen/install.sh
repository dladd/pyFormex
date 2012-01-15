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
_usage() {
    cat <<EOF
This script helps with installing tetgen from source. The normal sequence
of commands is:

./tetgen_install get unpack make
sudo ./tetgen_install install rename
./tetgen_install clean

Use at your own risk if you do not understand what is happening!

EOF
}

VERSION=1.4.3
NAME=tetgen$VERSION
ARCHIVE=$NAME.tar.gz
URI=http://tetgen.org/files/$ARCHIVE

_get() {
    [ -f $ARCHIVE ] || wget $URI
}

_unpack() {
    rm -rf $NAME
    tar xvzf $ARCHIVE
}

_make() {
    pushd $NAME
    CXXFLAGS=-O2 make -e
    popd
}

_install() {
    [ "$EUID" == "0" ] || {
	echo "install should be done as root!"
	return
    }
    pushd $NAME
    /usr/bin/install -m0755 tetgen /usr/local/bin
    popd
}


_clean() {
    rm -rf $NAME
    rm -f $ARCHIVE
}

for cmd in "$@"; do

    case $cmd in 
	get | unpack | make | install | rename | clean | usage ) _$cmd;;
	all ) _get;_unpack;_make;_install;_clean;;
        * ) _usage; echo "UNKNOWN command $cmd";;
    esac

done
