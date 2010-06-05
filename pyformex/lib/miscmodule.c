/* $Id$ */
//
//  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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

#include <Python.h>
#include <numpy/arrayobject.h>

static char __doc__[] = "_misc module\n\
\n\
This module provides accelerated versions of some pyFormex functions.\n\
\n";

/**************************************************** isclose ****/
/* Check if two floats are equal within a given tolerance */
/* int isclose(float a, float b, float rtol, float atol) */
/* { */
/*   printf("Compare %e %e %e %e\n",a,b,rtol,atol);  */
/*   int ok; */
/*   ok = fabs(a-b) < atol + rtol * fabs(b); */
/*   printf("  a-b: %e; atol+rtol*b %e\n",fabs(a-b),atol + rtol * fabs(b)); */
/*   printf("  ok: %d\n",ok); */
/*   return ok; */
/* } */


/**************************************************** coords.fuse ****/
/* Fuse nodes : new and much faster algorithm */
/* args:  x, val, flag, sel, tol
     x   : (nnod,3) coordinates
     val : (nnod) gives the point a code, such that only points with equal val
         are close. points in x are sorted according to val.
     flag: (nnod) initially 1, set to 0 if point is fused with another.
     sel : (nnod) 
     tol : tolerance for defining equality of coordinates
*/  
static PyObject * coords_fuse2(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *x;
  int *flag;
  int *val,*sel;
  float tol;
  if (!PyArg_ParseTuple(args, "OOOOf", &arg1, &arg2, &arg3, &arg4, &tol)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  arr3 = PyArray_FROM_OTF(arg3, NPY_INT, NPY_INOUT_ARRAY);
  if (arr3 == NULL) goto fail;
  arr4 = PyArray_FROM_OTF(arg4, NPY_INT, NPY_INOUT_ARRAY);
  if (arr4 == NULL) goto fail;
  /* We suppose the dimensions are correct*/
  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nnod;
  nnod = dims[0];
  x = (float *)PyArray_DATA(arr1);
  val = (int *)PyArray_DATA(arr2);
  flag = (int *)PyArray_DATA(arr3);
  sel = (int *)PyArray_DATA(arr4);
  int i,j,ki,kj,nexti;

  nexti = 1;
  for (i=1; i<nnod; i++) {
    j = i-1;
    ki = 3*i;
    while (j >= 0 && val[i]==val[j]) {
      kj = 3*j;
      if ( fabs(x[ki]-x[kj]) < tol &&
	   fabs(x[ki+1]-x[kj+1]) < tol &&
	   fabs(x[ki+2]-x[kj+2]) < tol ) {
	flag[i] = 0;
	sel[i] = sel[j];
	break;
      }
      --j;
    }
    if (flag[i]) {
      sel[i] = nexti;
      ++nexti;
    }
  }
  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  Py_DECREF(arr4);
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  printf("Error Cleanup\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  return NULL;
}


/**************************************************** coords.fuse ****/
/* Fuse nodes : new and much faster algorithm */
/* args:  x, val, flag, sel, tol
     x   : (nnod,3) coordinates
     val : (nnod) gives the point a code, such that only points with equal val
         are close. points in x are sorted according to val.
     flag: (nnod) initially 1, set to 0 if point is fused with another.
     sel : (nnod) 
     tol : tolerance for defining equality of coordinates
*/  
static PyObject * coords_fuse(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *x;
  int *flag;
  int *val,*sel;
  float tol;
  if (!PyArg_ParseTuple(args, "OOOOf", &arg1, &arg2, &arg3, &arg4, &tol)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  arr3 = PyArray_FROM_OTF(arg3, NPY_INT, NPY_INOUT_ARRAY);
  if (arr3 == NULL) goto fail;
  arr4 = PyArray_FROM_OTF(arg4, NPY_INT, NPY_INOUT_ARRAY);
  if (arr4 == NULL) goto fail;
  /* We suppose the dimensions are correct*/
  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nnod;
  nnod = dims[0];
  x = (float *)PyArray_DATA(arr1);
  val = (int *)PyArray_DATA(arr2);
  flag = (int *)PyArray_DATA(arr3);
  sel = (int *)PyArray_DATA(arr4);
  int i,j,ki,kj;

  for (i=0; i<nnod; i++) {
    ki = 3*i;
    j = i-1;
    while (j >= 0 && val[i]==val[j]) {
      kj = 3*j;
      if ( fabs(x[ki]-x[kj]) < tol && \
	   fabs(x[ki+1]-x[kj+1]) < tol && \
	   fabs(x[ki+2]-x[kj+2]) < tol ) {
	flag[i] = 0;
	sel[i] = sel[j];
	j = i+1;
	while (j < nnod) --sel[j++];
	break;
      }
      j = j-1;
    }
  }

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  Py_DECREF(arr4);
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  //printf("Error Cleanup\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  return NULL;
}


/**************************************************** nodal_sum ****/
/* Nodal sum of values defined on elements */
/* args:  val, elems, avg
    val   : (nelems,nplex,nval) values defined at points of elements.
    elems : (nelems,nplex) nodal ids of points of elements.
    work  : (elems.max()+1,nval) : workspace, should be zero on entry
    avg   : 0/1 
    The return value is a (nelems,nplex,nval) array where each value is
    replaced with the sum of its value at that node.
    If avg=True, the values are replaced with the average instead.
    (CURRENTLY NOT IMPLEMENTED)
    The operations are done in-place. The return value is None.
*/  
static PyObject * nodal_sum(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL;
  float *val,*work;
  int *elems;
  int avg;
  if (!PyArg_ParseTuple(args, "OOOf", &arg1, &arg2, &arg3, &avg)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  arr3 = PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr3 == NULL) goto fail;
  /* We suppose the dimensions are correct*/
  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nelems,nplex,nval;
  nelems = dims[0];
  nplex = dims[1];
  nval = dims[2];

  val = (float *)PyArray_DATA(arr1);
  elems = (int *)PyArray_DATA(arr2);
  work = (float *)PyArray_DATA(arr3);
  printf(" nelems=%d, nplex=%d, nval=%d\n",nelems,nplex,nval);
  
  int i,k,n;
  /* Loop over the input and sum */
  for (i=0;i<nelems*nplex;i++) {
    n = elems[i];
    for (k=0;k<3;k++) work[3*n+k] += val[3*i+k];
  }
  /* Place back results */
  for (i=0;i<nelems*nplex;i++) {
    n = elems[i];
    for (k=0;k<3;k++) val[3*i+k] = work[3*n+k];
  }

  /* Clean up and return */
  printf("Cleaning up\n");
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  printf("Error Cleanup\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  return NULL;
}


/**************************************************** nodal_sum_1 ****/
/* Nodal sum of values defined on elements */
/* args:  val, elems, nodes, avg
    val   : (nnod,nval) values defined at points of elements.
    elems : (nelems,nplex) nodal ids of points of elements.
    work  : (nnod,nval) : workspace, should be zero on entry
    avg   : 0/1 
    The return value is a (nnod,nval) with the sum of its value
    at that node.
    
    The operations are done in-place. The return value is None.
*/  
static PyObject * nodal_sum_1(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL;
  float *val,*work;
  int *elems,*nodes;
  int avg;
  if (!PyArg_ParseTuple(args, "OOOOf", &arg1, &arg2, &arg3, &arg4, &avg)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  arr3 = PyArray_FROM_OTF(arg3, NPY_INT, NPY_IN_ARRAY);
  if (arr3 == NULL) goto fail;
  arr4 = PyArray_FROM_OTF(arg4, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr4 == NULL) goto fail;
  /* We suppose the dimensions are correct*/
  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nelems,nplex,nval;
  nelems = dims[0];
  nplex = dims[1];
  nval = dims[2];
  dims = PyArray_DIMS(arg3);
  int nnod;
  nnod = dims[0];

  val = (float *)PyArray_DATA(arr1);
  elems = (int *)PyArray_DATA(arr2);
  nodes = (int *)PyArray_DATA(arr3);
  work = (float *)PyArray_DATA(arr4);
  printf(" nelems=%d, nplex=%d, nnod=%d\n",nelems,nplex,nnod);
  
  int i,k,n;
  /* Loop over the input and sum */
  for (i=0;i<nelems*nplex;i++) {
    n = elems[i];
    for (k=0;k<3;k++) work[3*n+k] += val[3*i+k];
  }
  /* Place back results */
  for (i=0;i<nelems*nplex;i++) {
    n = elems[i];
    for (k=0;k<3;k++) val[3*i+k] = work[3*n+k];
  }

  /* Clean up and return */
  printf("Cleaning up\n");
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  Py_DECREF(arr4);
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  printf("Error Cleanup\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  return NULL;
}


/* The methods defined in this module */
static PyMethodDef __methods__[] = {
    {"_fuse2", coords_fuse2, METH_VARARGS, "Fuse nodes."},
    {"_fuse", coords_fuse, METH_VARARGS, "Fuse nodes."},
    {"nodalSum", nodal_sum, METH_VARARGS, "Nodal sum."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module */
PyMODINIT_FUNC initmisc(void)
{
  (void) Py_InitModule3("misc", __methods__, __doc__);
    import_array(); /* Get access to numpy array API */
}

/* End */
