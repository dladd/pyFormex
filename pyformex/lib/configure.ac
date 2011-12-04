# $Id$                                              -*- Autoconf -*-
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

#
# Process this file with autoconf to produce a configure script.
#

# AX_PYTHON_VER( )
# ----------------
# Determine the python version
# Set value in PYTHON_VER
AC_DEFUN([AX_PYTHON_VER],
[
    AC_MSG_CHECKING(for Python version)
    ax_python_ver=[python$(python -c "import sys;print sys.version[:3]")]
    AC_SUBST([PYTHON_VER], ["${ax_python_ver}"])
    AC_MSG_RESULT([$ax_python_ver])
])

# AX_PYTHON_INCLUDE( )
# --------------------
# Determine the python include dir
# Set value in PYTHON_INCLUDE
AC_DEFUN([AX_PYTHON_INCLUDE],
[
    AC_MSG_CHECKING(for Python include dir)
    ax_python_include=[$(python configure_py -P)]
    AC_SUBST([PYTHON_INCLUDE], ["${ax_python_include}"])
    AC_MSG_RESULT([$ax_python_include])
])


# AX_NUMPY_INCLUDE( )
# -------------------
# Determine the numpy include path
# Set value in NUMPY_INCLUDE

AC_DEFUN([AX_NUMPY_INCLUDE],
[
    AC_MSG_CHECKING(for Numpy include dir)
    ax_numpy_include=[$(python configure_py -N)]
    AC_SUBST([NUMPY_INCLUDE], ["${ax_numpy_include}"])
    AC_MSG_RESULT([$ax_numpy_include])
])


# AX_SUCCESS( )
# -------------
# Determine success of previous tests
# Set value in SUCCESS

AC_DEFUN([AX_SUCCESS],
[
    ax_success=1
    if test x$PYTHON_INCLUDE = x; then
        ax_success=0
    fi	
    if test x$NUMPY_INCLUDE = x; then
        ax_success=0
    fi	
    AC_SUBST([SUCCESS], ["${ax_success}"])
    AC_MSG_RESULT([Success: $ax_success])
])

# AX_FIND_SOURCE( )
# -----------------
# Determine the C source files to compile
# Set value in C_SOURCE

AC_DEFUN([AX_C_SOURCE],
[
    AC_MSG_CHECKING(for C sourcefiles)
    ax_c_source=[$(ls -x *module.c|sed 's|$|\\|')]
    AC_SUBST([C_SOURCE], ["${ax_c_source}"])
    AC_MSG_RESULT([$ax_c_source])
])


AC_PREREQ(2.60)
AC_INIT(pyformex-lib,0.8.5)
AC_PROG_CC

AC_CHECK_HEADERS([GL/gl.h], [], [], [])
AX_PYTHON_INCLUDE
AX_NUMPY_INCLUDE
AX_C_SOURCE

AX_SUCCESS

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
