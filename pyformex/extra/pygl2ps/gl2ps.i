/* $Id$        -*- C -*-  */
//
//  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
//  pyFormex is a tool for generating, manipulating and transforming 3D
//  geometrical models by sequences of mathematical operations.
//  Home page: http://pyformex.org
//  Project page:  http://savannah.nongnu.org/projects/pyformex/
//  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
//  Distributed under the GNU General Public License version 3 or later.
//
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see http://www.gnu.org/licenses/.
//

/*
  Adapted for gl2ps-1.3.3 by Benedict Verhegghe
  Original by Toby White and Lothar Birk (Jan 2004)
*/
%module gl2ps
%{
#include "gl2ps.h"
%}


typedef int     GLint;
typedef short   GLshort;
typedef float   GLfloat;
typedef GLfloat GL2PSrgba[4];


%typemap(in) FILE* {
  $1 = (FILE *) PyFile_AsFile($input);
  printf("BV: Received a file\n");
}


/* Version number */

#define GL2PS_MAJOR_VERSION 1
#define GL2PS_MINOR_VERSION 3
#define GL2PS_PATCH_VERSION 3
#define GL2PS_EXTRA_VERSION ""

#define GL2PS_VERSION (GL2PS_MAJOR_VERSION + \
                       0.01 * GL2PS_MINOR_VERSION + \
                       0.0001 * GL2PS_PATCH_VERSION)

#define GL2PS_COPYRIGHT "(C) 1999-2009 C. Geuzaine"

/* Output file formats (the values and the ordering are important!) */

#define GL2PS_PS  0
#define GL2PS_EPS 1
#define GL2PS_TEX 2
#define GL2PS_PDF 3
#define GL2PS_SVG 4
#define GL2PS_PGF 5

/* Sorting algorithms */

#define GL2PS_NO_SORT     1
#define GL2PS_SIMPLE_SORT 2
#define GL2PS_BSP_SORT    3

/* Message levels and error codes */

#define GL2PS_SUCCESS       0
#define GL2PS_INFO          1
#define GL2PS_WARNING       2
#define GL2PS_ERROR         3
#define GL2PS_NO_FEEDBACK   4
#define GL2PS_OVERFLOW      5
#define GL2PS_UNINITIALIZED 6

/* Options for gl2psBeginPage */

#define GL2PS_NONE                 0
#define GL2PS_DRAW_BACKGROUND      (1<<0)
#define GL2PS_SIMPLE_LINE_OFFSET   (1<<1)
#define GL2PS_SILENT               (1<<2)
#define GL2PS_BEST_ROOT            (1<<3)
#define GL2PS_OCCLUSION_CULL       (1<<4)
#define GL2PS_NO_TEXT              (1<<5)
#define GL2PS_LANDSCAPE            (1<<6)
#define GL2PS_NO_PS3_SHADING       (1<<7)
#define GL2PS_NO_PIXMAP            (1<<8)
#define GL2PS_USE_CURRENT_VIEWPORT (1<<9)
#define GL2PS_COMPRESS             (1<<10)
#define GL2PS_NO_BLENDING          (1<<11)
#define GL2PS_TIGHT_BOUNDING_BOX   (1<<12)

/* Arguments for gl2psEnable/gl2psDisable */

#define GL2PS_POLYGON_OFFSET_FILL 1
#define GL2PS_POLYGON_BOUNDARY    2
#define GL2PS_LINE_STIPPLE        3
#define GL2PS_BLEND               4

/* Text alignment (o=raster position; default mode is BL):
   +---+ +---+ +---+ +---+ +---+ +---+ +-o-+ o---+ +---o 
   | o | o   | |   o |   | |   | |   | |   | |   | |   | 
   +---+ +---+ +---+ +-o-+ o---+ +---o +---+ +---+ +---+ 
    C     CL    CR    B     BL    BR    T     TL    TR */

#define GL2PS_TEXT_C  1
#define GL2PS_TEXT_CL 2
#define GL2PS_TEXT_CR 3
#define GL2PS_TEXT_B  4
#define GL2PS_TEXT_BL 5
#define GL2PS_TEXT_BR 6
#define GL2PS_TEXT_T  7
#define GL2PS_TEXT_TL 8
#define GL2PS_TEXT_TR 9

GLint gl2psBeginPage(const char *title, const char *producer,
		     GLint viewport[4], GLint format, GLint sort,
		     GLint options, GLint colormode,
		     GLint colorsize, GL2PSrgba *colormap,
		     GLint nr, GLint ng, GLint nb, GLint buffersize,
		     FILE* stream, const char *filename);

GLint gl2psEndPage(void);
GLint gl2psSetOptions(GLint options);
GLint gl2psGetOptions(GLint *options);
GLint gl2psBeginViewport(GLint viewport[4]);
GLint gl2psEndViewport(void);
GLint gl2psText(const char *str, const char *fontname, GLshort fontsize);
GLint gl2psTextOpt(const char *str, const char *fontname,
                                GLshort fontsize, GLint align, GLfloat angle);
GLint gl2psSpecial(GLint format, const char *str);
GLint gl2psDrawPixels(GLsizei width, GLsizei height,
		      GLint xorig, GLint yorig,
		      GLenum format, GLenum type, const void *pixels);
GLint gl2psEnable(GLint mode);
GLint gl2psDisable(GLint mode);
GLint gl2psPointSize(GLfloat value);
GLint gl2psLineWidth(GLfloat value);
GLint gl2psBlendFunc(GLenum sfactor, GLenum dfactor);
