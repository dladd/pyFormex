# Makefile for gl2ps
#
# $Id$
#
# (C) 2008 Benedict Verhegghe
#
# Makefile for gl2ps.
# The suggested use is:
#    make -f gl2ps.Makefile get             (to download the source)
#    make -f gl2ps.Makefile build           (to build the library) 
#    make -f gl2ps.Makefile install         (needs root permissions!)
#    make -f gl2ps.Makefile mrproper        (to clean up after install)
#
VERSION=1.3.2
GL2PS=gl2ps-${VERSION}
TARBALL=${GL2PS}.tgz
URL=http://geuz.org/gl2ps/src/${TARBALL}

PREFIX=/usr/local
LIBDIR=${PREFIX}/lib
LIBNAME=libgl2ps.so
LIBVER=${LIBNAME}.${VERSION}
FULLNAME=${LIBDIR}/${LIBVER}
SHORTNAME=${LIBDIR}/${LIBNAME}

BUILDDIR=${GL2PS}


.PHONY: all get unpack build install


all: get unpack build install

get: ${TARBALL}

unpack: ${BUILDDIR}

build: ${BUILDDIR}/${LIBNAME}

install: ${FULLNAME} ${SHORTNAME}



${TARBALL}:
	wget ${URL} 

${BUILDDIR}: ${TARBALL}
	tar xvzf $<
	cp gl2ps.Makefile $(BUILDDIR)/Makefile


${BUILDDIR}/${LIBNAME}: ${BUILDDIR}
	make -C ${BUILDDIR} ${LIBNAME}


${LIBNAME}: gl2ps.o
	gcc -shared -Wl,-soname,${LIBNAME} -o $@ $<

gl2ps.o: gl2ps.c gl2ps.h
	gcc -fPIC -g -c -Wall -O3 $<

${FULLNAME}: ${BUILDDIR}/${LIBNAME}
	install -m 0755 ${BUILDDIR}/${LIBNAME} $@

${SHORTNAME}: ${BUILDDIR}/${LIBNAME}
	ln -sfn ${LIBVER} $@


clean:
	cd ${BUILDDIR}; rm *.o *.so.*

mrproper:
	rm -rf ${BUILDDIR}
