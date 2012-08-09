# $Id$

# Makefile for pyFormex extras

DIRS = dxfparser gts postabq

# the sets of directories to do various things in
BUILDDIRS = $(DIRS:%=build-%)
INSTALLDIRS = $(DIRS:%=install-%)
MANDIRS = $(DIRS:%=man-%)
CLEANDIRS = $(DIRS:%=clean-%)

build: $(BUILDDIRS)
$(DIRS): $(BUILDDIRS)
$(BUILDDIRS):
	$(MAKE) -C $(@:build-%=%) build

install: build $(INSTALLDIRS)
$(INSTALLDIRS):
	$(MAKE) -C $(@:install-%=%) install

manpages: $(MANDIRS)
$(MANDIRS):
	$(MAKE) -C $(@:man-%=%) manpages

clean: $(CLEANDIRS)
$(CLEANDIRS): 
	$(MAKE) -C $(@:clean-%=%) clean

all: build install clean

.PHONY: $(DIRS)
.PHONY: $(BUILDDIRS)
.PHONY: $(INSTALLDIRS)
.PHONY: $(CLEANDIRS)
.PHONY: all build install manpages clean
