# $Id$

#
# needs packages:  libboost-python-dev libftgl-dev
#
PYFTGL=PyFTGL-0.5c
PYFTGL_TGZ=$(PYFTGL).tar.bz2
PYFTGL_URL=http://pyftgl.googlecode.com/files/$(PYFTGL_TGZ)
SRCDIR=pyftgl

.PHONY: all download unpack build install clean

all: build 

download: $(PYFTGL_TGZ)

$(PYFTGL_TGZ):
	wget $(PYFTGL_URL)

unpack: $(PYFTGL)

$(PYFTGL): $(PYFTGL_TGZ)
	tar xvjf $<

build: $(PYFTGL)
	cd $(SRCDIR); python setup.py build

install:
	cd $(SRCDIR); python setup.py install

clean:
	rm -rf $(PYFTGL)
