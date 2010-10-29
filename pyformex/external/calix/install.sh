#!/bin/bash
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
