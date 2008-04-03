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

/****** EXTERNAL FUNCTIONS (callable from Python ********/

/********************************************** drawgl.draw_lines ****/
/* Draw a collection of lines. */
/* args:  x,c
    x : float (nels,2,3) : coordinates.
    c : float (nels,3) or (nels,2,3) : color(s)
*/  
static PyObject *
draw_lines(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL;
  PyObject *arr1=NULL, *arr2=NULL;
  float *x, *c=NULL;
  int nels,nd=0;

  //printf("** draw_lines\n");

  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;

  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  x = (float *)PyArray_DATA(arr1);
  nels = PyArray_DIMS(arr1)[0];

  arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr2 != NULL) {
    c = (float *)PyArray_DATA(arr2);
    nd = PyArray_NDIM(arr2);
  }

  glBegin(GL_LINES);
  int i;
  if (nd == 0) {
    //printf("** Draw without color\n");
    for (i=0; i<2*3*nels; i+=3) {
      //printf("coordinate %d\n",i);
      glVertex3fv(x+i);
    }
  } else if (nd == 2) {
    //printf("** Draw with 1 color\n");
    for (i=0; i<3*nels; i+=3) {
      glColor3fv(c+i);
      glVertex3fv(x+2*i);
      glVertex3fv(x+2*i+3);
    }
  } else if (nd == 3) {
    //printf("** Draw with 2 colors\n");
    for (i=0; i<2*3*nels; i+=3) {
      glColor3fv(c+i);
      glVertex3fv(x+i);
    }
  }
  glEnd();

  /* Cleanup */
  //printf("** Cleanup\n");
  Py_DECREF(arr1);
  if (arr2 != NULL) { Py_DECREF(arr2); }
  Py_INCREF(Py_None);
  return Py_None;
}


/********************************************** drawgl.draw_polygons ****/
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
  if (nplex < 3) goto cleanup;

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
  if (nplex == 3)
    glBegin(GL_TRIANGLES);
  else if (nplex == 4)
    glBegin(GL_QUADS);
  else
    glBegin(GL_POLYGON);

  int i,j;
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
	/*gl_color(c+3*i,alpha);*/
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

 cleanup:
  Py_DECREF(arr1);
  if (arr2 != NULL)  { Py_DECREF(arr2); }
  if (arr3 != NULL)  { Py_DECREF(arr3); }
  Py_INCREF(Py_None);
  return Py_None;
}


/***************************************** drawgl.draw_triangle_elements ****/
/* Draw triangle elements */
/* args:  x, n, c
     x  : float32 (npts,3) coordinates
     e  : int32 (ntri,3) elements
     n  : float32 (ntri,3) normals
     c  : float32 (3) or (ntri,3) or (ntri,3,3) colors
*/  
static PyObject *
draw_triangle_elements(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *x,*n=NULL,*c=NULL;
  int *e;

  if (debug) printf("** draw_triangle_elements\n");
  if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  x = (float *)PyArray_DATA(arr1);
  if (debug) printf("Got arg 1\n");

  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  e = (int *)PyArray_DATA(arr2);
  if (debug) printf("Got arg 2\n");

  arr3 = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr3 != NULL) 
    n = (float *)PyArray_DATA(arr3);
  if (debug) printf("Got arg 3\n");

  arr4 = PyArray_FROM_OTF(arg4, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr4 != NULL) 
    c = (float *)PyArray_DATA(arr4);
  if (debug) printf("Got arg 4\n");
  
  int npts,ntri;
  npy_intp * dims;
  dims = PyArray_DIMS(arr1);
  npts = dims[0];
  dims = PyArray_DIMS(arr2);
  ntri = dims[0];

  if (debug) printf("ntri = %d\n",ntri);
  glBegin(GL_TRIANGLES);
  int i,j;
  for (i=0; i<3*ntri; i+=3) {
    for (j=0; j<3; j+=3)
      glVertex3fv(x+3*e[i+j]);
  }
  glEnd();

  /* Cleanup */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  if (arr3 != NULL)  { Py_DECREF(arr3); }
  if (arr4 != NULL)  { Py_DECREF(arr4); }
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  if (debug) printf("Error Cleanup\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}


/***************** The methods defined in this module **************/
static PyMethodDef Methods[] = {
    {"drawLines", draw_lines, METH_VARARGS, "Draw lines."},
    {"drawPolygons", draw_polygons, METH_VARARGS, "Draw polygons."},
    {"drawTriangleElems", draw_triangle_elements, METH_VARARGS, "Draw triangle elements."},
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
