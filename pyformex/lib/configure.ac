#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.60)
AC_INIT(pyformex-lib,0.7-a2)
AC_PROG_CC

AC_CHECK_HEADERS([GL/gl.h], [], [], [])
AX_PYTHON_INCLUDE
AX_NUMPY_INCLUDE
AX_C_SOURCE

AX_SUCCESS

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
