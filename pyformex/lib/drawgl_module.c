/* $Id$ */
//
//  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
//  pyFormex is a tool for generating, manipulating and transforming 3D
//  geometrical models by sequences of mathematical operations.
//  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
//  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
    n = (float *)PyArray_DATA(arr2);
    ndn = PyArray_NDIM(arr2);
  }

  arr3 = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr3 != NULL) { 
    c = (float *)PyArray_DATA(arr3);
    ndc = PyArray_NDIM(arr3);
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

  /* Cleanup */
  Py_DECREF(arr1);
  if (arr2 != NULL)  { Py_DECREF(arr2); }
  if (arr3 != NULL)  { Py_DECREF(arr3); }
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
  if (arr1 == NULL) goto cleanup;
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

 cleanup:
  if (arr1 != NULL) { Py_DECREF(arr1); }
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
  if (arr1 == NULL) goto cleanup;
  x = (float *)PyArray_DATA(arr1);
  npts = PyArray_DIMS(arr1)[0];
#ifdef DEBUG
  printf("** npts = %d\n",npts);
#endif

  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto cleanup;
  e = (int *)PyArray_DATA(arr2);
  nel = PyArray_DIMS(arr2)[0];
  nplex = PyArray_DIMS(arr2)[1];
#ifdef DEBUG
  printf("** nel = %d\n",nel);
  printf("** nplex = %d\n",nplex);
#endif

  arr3 = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr3 != NULL) { 
    n = (float *)PyArray_DATA(arr3);
    ndn = PyArray_NDIM(arr3);
  }

  arr4 = PyArray_FROM_OTF(arg4, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr4 != NULL) { 
    c = (float *)PyArray_DATA(arr4);
    ndc = PyArray_NDIM(arr4);
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
#ifdef DEBUG
	printf("** check 1\n");
#endif
	for (i=0; i<nel*nplex; i++) {
	  gl_color(c+3*i,alpha);
	  glVertex3fv(x+3*e[i]);
	}
      } else if (ndn == 2) {   // DONE
#ifdef DEBUG
	printf("** check 2\n");
#endif
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
#ifdef DEBUG
	printf("** check 5\n");
#endif
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
#ifdef DEBUG
	printf("** check 7\n");
#endif
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  gl_color(c+3*i,alpha);
	  for (j=0;j<nplex;++j) glVertex3fv(x+3*e[nplex*i+j]);
	  glEnd();
	}
      } else if (ndn == 2){   // DONE
#ifdef DEBUG
	printf("** check 8\n");
#endif
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
#ifdef DEBUG
	printf("** check 10 1\n");
#endif
	for (i=0; i<nel; i++) {
	  glBegin(objtype);
	  for (j=0;j<nplex;++j) {
	    gl_color(c+3*(nplex*i+j),alpha);
	    glVertex3fv(x+3*e[nplex*i+j]);
	  }
	  glEnd();
	}
      } else if (ndn == 2) {   // DONE
#ifdef DEBUG
	printf("** check 11 2\n");
#endif
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
  
 cleanup:
  if (arr1 != NULL) { Py_DECREF(arr1); }
  if (arr2 != NULL) { Py_DECREF(arr2); }
  if (arr3 != NULL) { Py_DECREF(arr3); }
  if (arr4 != NULL) { Py_DECREF(arr4); }
  Py_INCREF(Py_None);
  return Py_None;
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
  if (arr1 == NULL) goto cleanup;
  x = (float *)PyArray_DATA(arr1);
  npts = PyArray_DIMS(arr1)[0];
  printf("** npts = %d\n",npts);

  arr2 = PyArray_FROM_OTF(arg2,NPY_INT,NPY_IN_ARRAY);
  if (arr2 == NULL) goto cleanup;
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

 cleanup:
  if (arr1 != NULL) { Py_DECREF(arr1); }
  if (arr2 != NULL) { Py_DECREF(arr2); }
  Py_INCREF(Py_None);
  return Py_None;
}


/***************** The methods defined in this module **************/
static PyMethodDef _methods_[] = {
    {"get_version", get_version, METH_VARARGS, "Return library version."},
    {"draw_polygons", draw_polygons, METH_VARARGS, "Draw polygons."},
    {"pick_polygons", pick_polygons, METH_VARARGS, "Pick polygons."},
    {"draw_polygon_elems", draw_polygon_elems, METH_VARARGS, "Draw polygon elements."},
    {"pick_polygon_elems", pick_polygon_elems, METH_VARARGS, "Pick polygon elements."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module */
PyMODINIT_FUNC initdrawgl_(void)
{
  (void) Py_InitModule3("drawgl_", _methods_, __doc__);
  import_array(); /* Get access to numpy array API */
}

/* End */
