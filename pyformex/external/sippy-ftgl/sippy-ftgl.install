#!/bin/bash
# $Id$
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
