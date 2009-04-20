#!/bin/bash
# $Id$
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
