/* $Id$ */
//
//  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
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
  Low level drawing functions to speed up OpenGL calls on large arrays.
 
  The callers should make sure that the arguments are correct.
  Nasty crashes may result if not!
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <GL/gl.h>
#include <GL/glu.h>

static char __doc__[] = "drawgl_ module\n\
\n\
This module provides accelerated versions of the pyFormex basic\n\
OpenGL drawing functions.\n\
\n";


/************************ LIBRARY VERSION *******************/
/*
  Whenever a change is made to this library that causes pyFormex
  to be incompatible with the previous version, the version number
  should be bumped, and the new version number should also be set
  in the lib module initialization file __init__.py

 ## ACTUALLY NOT USED YET ##
*/

int version = 1;

static PyObject *
get_version(PyObject *dummy, PyObject *args) 
{
  return Py_BuildValue("i", version);
}


/****** INTERNAL FUNCTIONS (not callable from Python ********/

/********************************************** gl_color ****/
/* Set the OpenGL color, possibly with transparency. */
/*
    color is an array of 3 float values.
    alpha is a single float value.
    All values are between 0.0 and 1.0
*/
void gl_color(float *color, float alpha)
{
  if (alpha == 1.0) {
    glColor3fv(color);
  } else {
    glColor4f(color[0],color[1],color[2],alpha);
  } 
  return;
}

/********************************************** gl_objtype ****/
/* Set the OpenGL object type from plexitude. */
/*
    nplex is an int >= 0.
*/
int gl_objtype(int nplex)
{
  int objtype;
  if (nplex == 1)
    objtype = GL_POINTS;
  else if (nplex == 2)
    objtype = GL_LINES;
  else if (nplex == 3)
    objtype = GL_TRIANGLES;
  else if (nplex == 4)
    objtype = GL_QUADS;
  else
    objtype = GL_POLYGON;
  return objtype;
}

/********************************************** gl_map2_vertexmode ****/
/* Set the OpenGL MAP2 vertex mode from ndim. */
/*
    ndim is either 3 or 4.
*/
GLenum gl_map2_vertexmode(int ndim)
{
  GLenum mode = 0;
  if (ndim == 3)
    mode = GL_MAP2_VERTEX_3;
  else if (ndim == 4)
    mode = GL_MAP2_VERTEX_4;
  return mode;
}

/****** EXTERNAL FUNCTIONS (callable from Python ********/

/********************************************** draw_polygons ****/
/* Draw polygons */
/* args:  x
    x : float (nel,nplex,3) : coordinates
    n : float (nel,3) or (nel,nplex,3) normals.
    c : float (nel,3) or (nel,nplex,3) colors
    alpha : float
    objtype : GL Object type (-1 = auto)
*/  
static PyObject *
draw_polygons(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL;
  float *x, *n=NULL, *c=NULL, alpha;
  int objtype,nel,nplex,ndc=0,ndn=0,i,j;

#ifdef DEBUG
  printf("** draw_polygons\n");
#endif

  if (!PyArg_ParseTuple(args,"OOOfi",&arg1,&arg2,&arg3,&alpha,&objtype)) return NULL;

  arr1 = PyArray_FROM_OTF(arg1,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  x = (float *)PyArray_DATA(arr1);
  nel = PyArray_DIMS(arr1)[0];
  nplex = PyArray_DIMS(arr1)[1];
#ifdef DEBUG
  printf("** nel = %d\n",nel);
  printf("** nplex = %d\n",nplex);
#endif

  arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr2 != NULL) { 
    ndn = PyArray_NDIM(arr2);
    n = (float *)PyArray_DATA(arr2);
  }

  arr3 = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr3 != NULL) { 
    ndc = PyArray_NDIM(arr3);
    c = (float *)PyArray_DATA(arr3);
  }
#ifdef DEBUG
  printf("** ndn = %d\n",ndn);
  printf("** ndc = %d\n",ndc);
#endif
  
  if (objtype < 0) objtype = gl_objtype(nplex);

  if (nplex <= 4 && objtype == gl_objtype(nplex)) { 
    /*********** Points, Lines, Triangles, Quads **************/
    glBegin(objtype);

    if (ndc < 2) {        /* no or single color */
      if (ndc == 1) {    	/* single color */
	gl_color(c,alpha);
      }
      if (ndn == 0) {
	for (i=0; i<nel*nplex*3; i+=3) {
	  glVertex3fv(x+i);
	}
      } else if (ndn == 2) {
	for (i=0; i<nel; i++) {
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	}
      } else if (ndn == 3) {
	for (j=0;j<nel*nplex*3;j+=3) {
	  glNormal3fv(n+j);
	  glVertex3fv(x+j);
	}
      }
    } else if (ndc == 2) {     /* element color */
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	}
      } else if (ndn == 2){
	for (i=0; i<nel; i++) {
	  gl_color(c+3*i,alpha);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex*3;j+=3) {
	    glNormal3fv(n+nplex*3*i+j);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	}
      }
    } else if (ndc == 3) {     /* vertex color */
      if (ndn == 0) {
	for (i=0; i<nel*nplex*3; i+=3) {
	  gl_color(c+i,alpha);
	  glVertex3fv(x+i);
	}
      } else if (ndn == 2) {
	for (i=0; i<nel; i++) {
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) {
	    gl_color(c+nplex*3*i+j,alpha);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	}
      } else if (ndn == 3) { 
	for (i=0; i<nel; i++) {
	  for (j=0;j<nplex*3;j+=3) {
	    glNormal3fv(n+nplex*3*i+j);
	    gl_color(c+nplex*3*i+j,alpha);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	}
      }
    }
    glEnd();

  } else {
    /************** Polygons ********************/

    if (ndc < 2) {        /* no or single color */
      if (ndc == 1) {    	/* single color */
	gl_color(c,alpha);
      }
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	  glEnd();
	}
      } else if (ndn == 2) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	  glEnd();
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex*3;j+=3) {
	    glNormal3fv(n+nplex*3*i+j);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	  glEnd();
	}
      }
    } else if (ndc == 2) {    /* element color */
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	  glEnd();
	}
      } else if (ndn == 2){
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
	  glEnd();
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex*3;j+=3) {
	    glNormal3fv(n+nplex*3*i+j);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	  glEnd();
	}
      }
    } else if (ndc == 3) {         /* vertex color */
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex*3;j+=3) {
	    gl_color(c+nplex*3*i+j,alpha);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	  glEnd();
	}
      } else if (ndn == 2) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) {
	    gl_color(c+nplex*3*i+j,alpha);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	  glEnd();
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex*3;j+=3) {
	    glNormal3fv(n+nplex*3*i+j);
	    gl_color(c+nplex*3*i+j,alpha);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	  glEnd();
	}
      }
    }
  }

  /* cleanup: */
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_INCREF(Py_None);
  return Py_None;
}


/********************************************** pick_polygons ****/
/* Pick polygons */
/* args: 
    x : float (nel,nplex,3) : coordinates
    objtype : GL Object type (-1 = auto)
*/  
static PyObject *
pick_polygons(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL;
  PyObject *arr1=NULL;
  float *x;
  int objtype,nel,nplex,i,j;

#ifdef DEBUG
  printf("** pick_polygons\n");
#endif
  if (!PyArg_ParseTuple(args,"Oi",&arg1,&objtype)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  x = (float *)PyArray_DATA(arr1);
  nel = PyArray_DIMS(arr1)[0];
  nplex = PyArray_DIMS(arr1)[1];
#ifdef DEBUG
  printf("** nel = %d\n",nel);
  printf("** nplex = %d\n",nplex);
#endif

  if (objtype < 0) objtype = gl_objtype(nplex);
  for (i=0; i<nel; i++) {
    glPushName(i);
    glBegin(objtype);
    for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
#ifdef DEBUG
  printf("** point %d: ",i);
  int k;
  for (j=0;j<nplex*3;j+=3) for (k=j;k<j+3;++k) printf(" %10.3e,",x[nplex*3*i+k]);
  printf("\n");
#endif
    glEnd();
    glPopName();
  }

  /* cleanup: */
  Py_XDECREF(arr1);
  Py_INCREF(Py_None);
  return Py_None;
}


/********************************************** draw_polygon_elements ****/
/* Draw polygon elements */
/* args:  x
    x : float (npts,3) : coordinates
    e : int32 (nel,nplex) : element connectivity
    n : float (nel,3) or (nel,nplex,3) normals.
    c : float (nel,3) or (nel,nplex,3) colors
    alpha : float
    objtype : GL Object type (-1 = auto)
*/  
static PyObject *
draw_polygon_elems(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *x, *n=NULL, *c=NULL, alpha;
  int *e, objtype;
  int npts,nel,nplex,ndc=0,ndn=0,i,j;

#ifdef DEBUG
  printf("** draw_polygon_elements\n");
#endif

  if (!PyArg_ParseTuple(args,"OOOOfi",&arg1,&arg2,&arg3,&arg4,&alpha,&objtype)) return NULL;

  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) goto fail;
  x = (float *)PyArray_DATA(arr1);
  npts = PyArray_DIMS(arr1)[0];
#ifdef DEBUG
  printf("** npts = %d\n",npts);
#endif

  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  e = (int *)PyArray_DATA(arr2);
  nel = PyArray_DIMS(arr2)[0];
  nplex = PyArray_DIMS(arr2)[1];
#ifdef DEBUG
  printf("** nel = %d\n",nel);
  printf("** nplex = %d\n",nplex);
#endif

  arr3 = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr3 != NULL) { 
    ndn = PyArray_NDIM(arr3);
    n = (float *)PyArray_DATA(arr3);
  }

  arr4 = PyArray_FROM_OTF(arg4, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr4 != NULL) { 
    ndc = PyArray_NDIM(arr4);
    c = (float *)PyArray_DATA(arr4);
  }
  
#ifdef DEBUG
  printf("** ndn = %d\n",ndn);
  printf("** ndc = %d\n",ndc);
#endif
  
  if (objtype < 0) objtype = gl_objtype(nplex);

  if (nplex <= 4 && objtype == gl_objtype(nplex)) { 
    glBegin(objtype);

    if (ndc < 2) {        /* no or single color */
      if (ndc == 1) {    	/* single color */
	gl_color(c,alpha);
      }
      if (ndn == 0) {
	for (i=0; i<nel*nplex; ++i) {
	  glVertex3fv(x+3*e[i]);
	}
      } else if (ndn == 2) {
	for (i=0; i<nel; i++) {
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex; ++j) glVertex3fv(x+3*e[nplex*i+j]);
	}
      } else if (ndn == 3) {
	for (j=0;j<nel*nplex;++j) {
#if BUGFIX & 1
	  gl_color(c,alpha);
#endif
	  glNormal3fv(n+3*j);
	  glVertex3fv(x+3*e[j]);
	}
      }
    } else if (ndc == 2) {
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	}
      } else if (ndn == 2){
	for (i=0; i<nel; i++) {
	  gl_color(c+3*i,alpha);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex;++j) {
	    glNormal3fv(n+3*(nplex*i+j));
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	}
      }
    } else if (ndc == 3) {
      if (ndn == 0) {  // DONE
	for (i=0; i<nel*nplex; i++) {
	  gl_color(c+3*i,alpha);
	  glVertex3fv(x+3*e[i]);
	}
      } else if (ndn == 2) {   // DONE
	for (i=0; i<nel; i++) {
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex;++j) {
	    gl_color(c+3*(nplex*i+j),alpha);
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  for (j=0;j<nplex;++j) {
	    glNormal3fv(n+3*(nplex*i+j));
	    gl_color(c+3*(nplex*i+j),alpha);
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	}
      }
    }
    glEnd();

  } else {

#ifdef DEBUG
    printf("** objtype = %d\n",objtype);
#endif
    if (ndc < 2) {        /* no or single color */
      if (ndc == 1) {    	/* single color */
	gl_color(c,alpha);
      }
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	  glEnd();
	}
      } else if (ndn == 2) {   // DONE
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	  glEnd();
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex;++j) {
	    glNormal3fv(n+3*(nplex*i+j));
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	  glEnd();
	}
      }
    } else if (ndc == 2) {   // DONE
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	  glEnd();
	}
      } else if (ndn == 2){   // DONE
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	  glEnd();
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex;++j) {
	    glNormal3fv(n+3*(nplex*i+j));
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	  glEnd();
	}
      }
    } else if (ndc == 3) {   // DONE
      if (ndn == 0) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex;++j) {
	    gl_color(c+3*(nplex*i+j),alpha);
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	  glEnd();
	}
      } else if (ndn == 2) {   // DONE
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex;++j) {
	    gl_color(c+3*(nplex*i+j),alpha);
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	  glEnd();
	}
      } else if (ndn == 3) {
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex;++j) {
	    glNormal3fv(n+3*(nplex*i+j));
	    gl_color(c+3*(nplex*i+j),alpha);
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	  glEnd();
	}
      }
    }
  }
  
  /* cleanup: */
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  Py_INCREF(Py_None);
  return Py_None;
  
 fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  return NULL;
}


/********************************************** pick_polygon_elems ****/
/* Pick polygon elements */
/* args: 
    x : float (npts,3) : coordinates
    e : int32 (nel,nplex) : element connectivity
    objtype : GL Object type (-1 = auto)
*/  
static PyObject *
pick_polygon_elems(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL;
  PyObject *arr1=NULL, *arr2=NULL;
  float *x;
  int *e,objtype,npts,nel,nplex,i,j;

#ifdef DEBUG
  printf("** pick_polygon_elems\n");
#endif
  if (!PyArg_ParseTuple(args,"OOi",&arg1,&arg2,&objtype)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr1 == NULL) goto fail;
  x = (float *)PyArray_DATA(arr1);
  npts = PyArray_DIMS(arr1)[0];
#ifdef DEBUG
  printf("** npts = %d\n",npts);
#endif

  arr2 = PyArray_FROM_OTF(arg2,NPY_INT,NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  e = (int *)PyArray_DATA(arr2);
  nel = PyArray_DIMS(arr2)[0];
  nplex = PyArray_DIMS(arr2)[1];
#ifdef DEBUG
  printf("** nel = %d\n",nel);
  printf("** nplex = %d\n",nplex);
#endif

  if (objtype < 0) objtype = gl_objtype(nplex);
  for (i=0; i<nel; i++) {
    glPushName(i);
    glBegin(objtype);
    for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
    glEnd();
    glPopName();
  }

  /* cleanup: */
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_INCREF(Py_None);
  return Py_None;

 fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}

/********************************************** draw_nurbs_surfaces ****/
/* Draw NURBS surfaces */
/* args: ndim is 3 or 4 (4th value is weight)
    coords: x: (nsurf,ns,nt,ndim)
    sknots: s: (nsknots) or (nsurf,nsknots)
    tknots: t: (ntknots) or (nsurf,ntknots)
    color:  c: None or (ndim) or (nsurf,ndim) or (nsurf,ns,nt,4) !!
    alpha: float
    sampling: float
*/  
static PyObject* draw_nurbs_surfaces(PyObject *dummy, PyObject *args)
{
  PyObject *retval=NULL;
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *x, *s=NULL, *t=NULL, *c=NULL, alpha, sampling;
  int nsurf,ns,nt,ndim,nds,nsknots,nsorder,ndt,ntknots,ntorder,ndc=0,ncdim=0,mode,cmode,i;
  GLUnurbs *nurb=NULL;

#ifdef DEBUG
  printf("** draw_nurbs_surfaces\n");
#endif
  
  if (!PyArg_ParseTuple(args,"OOOOff",&arg1,&arg2,&arg3,&arg4,&alpha,&sampling)) return NULL;

  arr1 = PyArray_FROM_OTF(arg1,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr1 == NULL) goto fail;
  x = (float *)PyArray_DATA(arr1);
  nsurf = PyArray_DIMS(arr1)[0];
  ns = PyArray_DIMS(arr1)[1];
  nt = PyArray_DIMS(arr1)[2];
  ndim = PyArray_DIMS(arr1)[3];
#ifdef DEBUG
  printf("** nsurf = %d\n",nsurf);
  printf("** ns = %d\n",ns);
  printf("** nt = %d\n",nt);
  printf("** ndim = %d\n",ndim);
#endif
  
  arr2 = PyArray_FROM_OTF(arg2,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail; 
  s = (float *)PyArray_DATA(arr2);
  nds = PyArray_NDIM(arr2);
  nsknots = PyArray_DIMS(arr2)[nds-1];
  nsorder = nsknots - ns;
#ifdef DEBUG
  printf("** nds = %d\n",nds);
  printf("** nsknots = %d\n",nsknots);
#endif
  
  arr3 = PyArray_FROM_OTF(arg3,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr3 == NULL) goto fail; 
  t = (float *)PyArray_DATA(arr3);
  ndt = PyArray_NDIM(arr3);
  ntknots = PyArray_DIMS(arr3)[ndt-1];
  ntorder = ntknots - nt;
#ifdef DEBUG
  printf("** ndt = %d\n",ndt);
  printf("** ntknots = %d\n",ntknots);
#endif
  
  arr4 = PyArray_FROM_OTF(arg4,NPY_FLOAT,NPY_IN_ARRAY);
  if (arr4 != NULL) { 
    ndc = PyArray_NDIM(arr4);
    if (ndc > 0) {
      ncdim = PyArray_DIMS(arr4)[ndc-1];
      c = (float *)PyArray_DATA(arr4);
    }
  }
#ifdef DEBUG
  printf("** ndc = %d\n",ndc);
  printf("** ncdim = %d\n",ncdim);
#endif

  nurb = gluNewNurbsRenderer();
#ifdef DEBUG
  printf("** nurb = %p\n",nurb);
#endif
  if (nurb == NULL) goto cleanup;
  gluNurbsProperty(nurb,GLU_SAMPLING_TOLERANCE,sampling);

  mode = gl_map2_vertexmode(ndim);
  if (ndc == 4) 
    cmode = GL_MAP2_COLOR_4;

#ifdef DEBUG
  printf("** nurbs mode = %d\n",mode);
  printf("** nurbs cmode = %d\n",cmode);
  printf("** nurbs order = %d, %d\n",nsorder,ntorder);
#endif

  if (ndc == 1) {    	/* single color */
    gl_color(c,alpha);
  }

  for (i=0; i<nsurf; ++i) {
#ifdef DEBUG
    printf("** nurbs surface %d,\n",i);
#endif
    if (ndc == 2) {    	/* element color */
      gl_color(c,alpha);
      c += ncdim;
    }
   
    gluBeginSurface(nurb);
    if (ndc == 4 && ncdim == 4) {     /* vertex color */
      gluNurbsSurface(nurb,ns,s,nt,t,ncdim,ns*ncdim,c,nsorder,ntorder,cmode);
      c += ns*nt*ncdim; 
    }
    gluNurbsSurface(nurb,nsknots,s,ntknots,t,ndim,ns*ndim,x,nsorder,ntorder,mode);
    gluEndSurface(nurb);

    if (nds > 1) s += nsknots;
    if (ndt > 1) t += ntknots;
    x += ns*nt*ndim;
  }

 cleanup:
  Py_INCREF(Py_None);
  retval = Py_None;

  /* common code for normal and failure exit */
 fail:
  if (nurb) gluDeleteNurbsRenderer(nurb);
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  return retval;
}


/***************** The methods defined in this module **************/
static PyMethodDef _methods_[] = {
    {"get_version", get_version, METH_VARARGS, "Return library version."},
    {"draw_polygons", draw_polygons, METH_VARARGS, "Draw polygons."},
    {"pick_polygons", pick_polygons, METH_VARARGS, "Pick polygons."},
    {"draw_polygon_elems", draw_polygon_elems, METH_VARARGS, "Draw polygon elements."},
    {"pick_polygon_elems", pick_polygon_elems, METH_VARARGS, "Pick polygon elements."},
    {"draw_nurbs_surfaces", draw_nurbs_surfaces, METH_VARARGS, "Draw NURBS surfaces."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module */
PyMODINIT_FUNC initdrawgl_(void)
{
  PyObject* module;
  module = Py_InitModule3("drawgl_", _methods_, __doc__);
  PyModule_AddIntConstant(module,"accelerated",1);
  import_array(); /* Get access to numpy array API */
}

/* End */
