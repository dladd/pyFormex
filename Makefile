# Makefile for pyformex
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#

############# SET THESE TO SUIT YOUR INSTALLATION ####################

# root of the installation tree: this is a reasonable default
ROOTDIR= /usr/local
# where to install pyformex: some prefer to use $(ROOTDIR) 
LIBDIR= $(ROOTDIR)/lib
# where to create symbolic links to the executable files
BINDIR= $(ROOTDIR)/bin
# where to install the documentation
DOCDIR= $(ROOTDIR)/share/doc
# where to install problem types for GiD: check that this is correct!
# comment this line if you do not want to install problem types

############# NOTHING CONFIGURABLE BELOW THIS LINE ###################

VERSION= 0.1.2
PYFORMEXDIR= pyformex-$(VERSION)
INSTDIR= $(LIBDIR)/$(PYFORMEXDIR)
DOCINSTDIR= $(DOCDIR)/$(PYFORMEXDIR)
PROGRAM= pyformex
SOURCE= formex.py canvas.py camera.py colors.py vector.py
HTMLDOCS= $(SOURCE:.py=.html)
DOCFILES= README COPYING History
EXAMPLES= examples/*.py
STAMPABLE= README History Makefile
NONSTAMPABLE= COPYING 
STAMP= ./Stamp 

.PHONY: install dist distclean

all:
	echo  "Do `make install' to install this program"


############ User installation ######################

install:
	install -d $(INSTDIR) $(BINDIR) $(DOCINSTDIR) $(DOCINSTDIR)/examples
	install -m 0664 $(SOURCE) $(INSTDIR)
	install -m 0775 $(PROGRAM) $(INSTDIR)
	install -m 0664 ${DOCFILES} $(DOCINSTDIR)
	install -m 0664 examples/* $(DOCINSTDIR)/examples
	ln -sfn $(INSTDIR)/$(PROGRAM) $(BINDIR)/$(PROGRAM)

remove:
	echo "There is no automatic installation procedure."""
	echo "Remove the entire pyformex directory from where you installed it."
	echo "Remove the symbolic link to the pyformex program."""
	echo "Remove the pyformex doc files."""

############ Creating Distribution ##################

dist:	dist.stamped

distdoc:
	mkdir -p doc/html
	for f in $(PROGRAM) $(SOURCE); do pydoc -w "./$f" ; done
	mv *.html doc/html

dist.stamped: distdoc
	make distclean
	mkdir $(PYFORMEXDIR) $(PYFORMEXDIR)/examples
	$(STAMP) -tStamp.template -oStamp.stamp
	$(STAMP) -tStamp.stamp -d$(PYFORMEXDIR) $(PROGRAM) $(SOURCE)
	$(STAMP) -tStamp.stamp -d$(PYFORMEXDIR)/examples $(EXAMPLES)
	$(STAMP) -tStamp.stamp -d$(PYFORMEXDIR) $(STAMPABLE)
	cp $(NONSTAMPABLE) $(PYFORMEXDIR)
	cp -R screenshots  $(PYFORMEXDIR)
	tar czf $(PYFORMEXDIR).tar.gz $(PYFORMEXDIR)


distclean:
	rm -rf $(PYFORMEXDIR)
	alldirs . "rm -f *~"

public: $(PYFORMEXDIR).tar.gz
	scp README $(PYFORMEXDIR).tar.gz mecatrix.ugent.be:/home/ftp/pub/pyformex
