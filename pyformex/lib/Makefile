#  -*- Makefile -*- for pyFormex Compiled Modules
# $Id: $


CSOURCE= $(wildcard *.c)
OBJECTS= $(CSOURCE:.c=.o)
SOBJECTS= $(CSOURCE:.c=.so)

CC= gcc

INC= -I/usr/include -I/usr/include/python2.4 \
     -I/usr/lib/python2.4/site-packages/numpy/core/include

CFLAGS= -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -DMAJOR_VERSION=1 -DMINOR_VERSION=0



all: $(SOBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $<

%.so: %.o
	$(CC) -shared $< -L/usr/lib -o $@

# End
