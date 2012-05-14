# $Id$
#
# Makefile for gts* commands used by pyFormex
#
# You need: libgts-dev, libglib-2.0-dev, pkg-config 
#
#
PREFIX= /usr/local
BINDIR= ${PREFIX}/bin

SRC= $(wildcard *.c)
BIN= $(addprefix gts, ${SRC:.c=})

FLAGS= `pkg-config --cflags --libs glib-2.0 gts`
CC= gcc
INSTALLBIN= install -m755 

.PHONY: all install

all: ${BIN}

gts%: %.c
	${CC} ${FLAGS} $< -o $@

install: ${BIN}
	${INSTALLBIN} ${BIN} ${BINDIR}

distclean:
	rm -f ${BIN}