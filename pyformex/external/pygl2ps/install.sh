#!/bin/bash
# $Id$
#
# Install Python wrapper for gl2ps
# If you change version, you need to adapt this file, gl2ps.i, setup.py
# and remove the gl2ps_wrap.c and gl2ps.py
#
[ "$1" = "all" ] || exit

GL2PS=gl2ps-1.3.3
GL2PS_TGZ=$GL2PS.tgz
GL2PS_URL=http://geuz.org/gl2ps/src/$GL2PS_TGZ


[ -f $GL2PS_TGZ ] || wget $GL2PS_URL
[ -f gl2ps_wrap.c ] || swig -python gl2ps.i
rm -rf $GL2PS
tar xvzf $GL2PS_TGZ
cp *.c *.py $GL2PS
pushd $GL2PS
python setup.py build
python setup.py install
popd
rm -rf $GL2PS
