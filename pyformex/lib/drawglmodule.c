/* $Id$ */

/*
  Low level drawing functions to speed up OpenGL calls on large arrays.
 
  The callers should make sure that the arguments are correct.
  Nasty crashes may result if not!
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <GL/gl.h>

int debug = 0;
int version = 1;


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
*/  
static PyObject *
draw_polygons(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL;
  float *x, *n=NULL, *c=NULL, alpha;
  int nel,nplex,ndc=0,ndn=0;

  if (debug) printf("** draw_polygons\n");
  if (!PyArg_ParseTuple(args, "OOOf", &arg1, &arg2, &arg3, &alpha)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  x = (float *)PyArray_DATA(arr1);
  nel = PyArray_DIMS(arr1)[0];
  nplex = PyArray_DIMS(arr1)[1];
  if (debug) printf("** nel = %d\n",nel);
  if (debug) printf("** nplex = %d\n",nplex);

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
  
  if (debug) printf("** ndn = %d\n",ndn);
  if (debug) printf("** ndc = %d\n",ndc);
  int i,j,objtype;
  
  objtype = gl_objtype(nplex);

  if (nplex <= 4) { 
    glBegin(objtype);

    if (ndc == 0) {
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
    } else if (ndc == 2) {
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
    } else if (ndc == 3) {
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

    if (ndc == 0) {
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
    } else if (ndc == 2) {
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
    } else if (ndc == 3) {
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
/* args:  x
    x : float (nel,nplex,3) : coordinates
*/  
static PyObject *
pick_polygons(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL;
  PyObject *arr1=NULL;
  float *x;
  int nel,nplex;
  int objtype;

  if (debug) printf("** pick_polygons\n");
  if (!PyArg_ParseTuple(args, "O", &arg1)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  x = (float *)PyArray_DATA(arr1);
  nel = PyArray_DIMS(arr1)[0];
  nplex = PyArray_DIMS(arr1)[1];
  if (debug) printf("** nel = %d\n",nel);
  if (debug) printf("** nplex = %d\n",nplex);

  objtype = gl_objtype(nplex);
  int i,j;
  for (i=0; i<nel; i++) {
    glPushName(i);
    glBegin(objtype);
    for (j=0;j<nplex*3;j+=3) glVertex3fv(x+nplex*3*i+j);
    glEnd();
    glPopName();
  }

  /* Cleanup */
  Py_DECREF(arr1);
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
*/  
static PyObject *
draw_polygon_elements(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *x, *n=NULL, *c=NULL, alpha;
  int *e;
  int npts,nel,nplex,ndc=0,ndn=0;

  if (debug) printf("** draw_polygon_elements\n");
  if (!PyArg_ParseTuple(args, "OOOOf", &arg1, &arg2, &arg3, &arg4, &alpha)) return NULL;

  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) goto cleanup;
  x = (float *)PyArray_DATA(arr1);
  npts = PyArray_DIMS(arr1)[0];
  if (debug) printf("** npts = %d\n",npts);

  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto cleanup;
  e = (int *)PyArray_DATA(arr2);
  nel = PyArray_DIMS(arr2)[0];
  nplex = PyArray_DIMS(arr2)[1];
  if (debug) printf("** nel = %d\n",nel);
  if (debug) printf("** nplex = %d\n",nplex);


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
  
  if (debug) printf("** ndn = %d\n",ndn);
  if (debug) printf("** ndc = %d\n",ndc);
  int i,j,objtype;
  
  objtype = gl_objtype(nplex);

  if (nplex <= 4) { 
    glBegin(objtype);

    if (ndc == 0) {
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
    } else if (ndc == 3) {   // TODO
      if (ndn == 0) {
	for (i=0; i<nel*nplex*3; i+=3) {
	  gl_color(c+i,alpha);
	  glVertex3fv(x+i);
	}
      } else if (ndn == 2) {   // TODO
	for (i=0; i<nel; i++) {
	  glNormal3fv(n+3*i);
	  for (j=0;j<nplex*3;j+=3) {
	    gl_color(c+nplex*3*i+j,alpha);
	    glVertex3fv(x+nplex*3*i+j);
	  }
	}
      } else if (ndn == 3) {   // TODO
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

  } else {   // TODO

    if (ndc == 0) {
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
    } else if (ndc == 2) {
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
    } else if (ndc == 3) {
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
  
 cleanup:
  if (arr1 != NULL) { Py_DECREF(arr1); }
  if (arr2 != NULL) { Py_DECREF(arr2); }
  if (arr3 != NULL) {Py_DECREF(arr3); }
  if (arr4 != NULL) {Py_DECREF(arr4); }
  Py_INCREF(Py_None);
  return Py_None;
}


/***************** The methods defined in this module **************/
static PyMethodDef Methods[] = {
    {"draw_polygons", draw_polygons, METH_VARARGS, "Draw polygons."},
    {"pick_polygons", pick_polygons, METH_VARARGS, "Pick polygons."},
    {"draw_polygon_elems", draw_polygon_elements, METH_VARARGS, "Draw polygon elements."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module */
PyMODINIT_FUNC
initdrawgl(void)
{
    (void) Py_InitModule("drawgl", Methods);
    import_array(); /* Get access to numpy array API */
}

/* End */
