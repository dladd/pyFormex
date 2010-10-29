#!/bin/bash
# $Id: install.sh 1583 2010-10-11 16:49:51Z bverheg $
#
# Install Python wrapper for ftgl
#
[ "$1" = "all" ] || exit

PYFTGL=PyFTGL-0.4b 
PYFTGL_TGZ=$PYFTGL.tar.gz
PYFTGL_URL=http://pyftgl.googlecode.com/files/$PYFTGL_TGZ

[ -f $PYFTGL_TGZ ] || wget $PYFTGL_URL
rm -rf $PYFTGL
tar xvzf $PYFTGL_TGZ
pushd $PYFTGL
python setup.py build
python setup.py install
popd
#rm -rf $PYFTGL
