#!/bin/bash
#
# This script helps with installing calpy from source:
#
# ./calpy_install get unpack make
# sudo ./calpy_install install
# ./calpy_install clean
#
# Use at your own risk if you do not understand what is happening!
#

VERSION=0.4-a5
NAME=calpy-$VERSION
ARCHIVE=$NAME.tar.gz
URI=ftp://bumps.ugent.be/pub/calpy/$ARCHIVE

_get() {
    [ -f $ARCHIVE ] || wget $URI
}

_unpack() {
    rm -rf $NAME
    tar xvzf $ARCHIVE
}

_make() {
    pushd $NAME
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
    popd
}


_clean() {
    rm -rf $NAME
    rm -f $ARCHIVE
}

for cmd in "$@"; do

    case $cmd in 
	get | unpack | patch | make | install | rename | clean ) _$cmd;;
	all ) _get;_unpack;_make;_install;_clean;;
        * ) echo "UNKNOWN command $cmd";;
    esac

done
