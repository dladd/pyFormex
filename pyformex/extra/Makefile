#
# Makefile for compiling Calculix on Debian GNU/Linux
#
# Adapted from the original Calculix Makefile by B. Verhegghe
# (benedict.verhegghe@ugent.be)
#
# The default Makefile provided with Calculix (www.dhondt.de) instructs
# you to compile the spooles and arpack libraries from sources.
# This adapted Makefile makes use of the spooles and arpack libraries
# as provided by Debian.
# To install these:  apt-get install libspooles-dev libarpack2-dev
#
# This is for Calculix 2.5
#

CFLAGS = -Wall -O -I /usr/include/spooles -DARCH="Linux" -DSPOOLES -DARPACK -DMATRIXSTORAGE
FFLAGS = -Wall -O -fopenmp

CC=cc
FC=gfortran

.c.o :
	$(CC) $(CFLAGS) -c $<
.f.o :
	$(FC) $(FFLAGS) -c $<

include Makefile.inc

SCCXMAIN = ccx_2.5.c

OCCXF = $(SCCXF:.f=.o)
OCCXC = $(SCCXC:.c=.o)
OCCXMAIN = $(SCCXMAIN:.c=.o)

LIBS =  -L/usr/lib \
	-lspooles \
	-larpack \
	-llapack \
	-lblas \
	-lpthread -lm -lc

ccx_2.5: $(OCCXMAIN) ccx_2.5.a
	./date.pl; $(CC) $(CFLAGS) -c ccx_2.5.c; $(FC) -Wall -O -o $@ $(OCCXMAIN) ccx_2.5.a $(LIBS)

ccx_2.5.a: $(OCCXF) $(OCCXC)
	ar vr $@ $?
                                                                               
