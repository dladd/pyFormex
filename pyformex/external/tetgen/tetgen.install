#!/bin/bash
# $Id$
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
URI=http://tetgen.berlios.de/files/$ARCHIVE

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
