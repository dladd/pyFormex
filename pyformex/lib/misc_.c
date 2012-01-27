/* $Id$ */
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

#include <Python.h>
#include <numpy/arrayobject.h>

static char __doc__[] = "misc_ module\n\
\n\
This module provides accelerated versions of miscellaneous pyFormex functions.\n\
\n";

/* Dot product of two vectors of length n */
/* ia and ib are the strides of the elements addressed starting from a, b */ 
static float dotprod(float *a, int ia, float *b, int ib, int n)
{
  int i;
  float t;
  t = 0.0;
  for (i=0; i<n; i++) {
    t += (*a)*(*b);
    a += ia;
    b += ib;
  }
  return t;
}

/**************************************************** tofile_float32 ****/
/* Write an array to file */
/*
   Use:  tofile_float32(data,file,format)
   The elements of a 2-D array are written in row order with the given format
   to the open file. After each row, a newline is written.
*/

static PyObject * tofile_float32(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL;
  PyObject *arr1=NULL;
  FILE *fp;
  char *fmt; /* single element format */

  if (!PyArg_ParseTuple(args, "OOs", &arg1, &arg2, &fmt)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  fp = PyFile_AsFile(arg2);
  if (!fp) goto fail;

  float *val;
  val = (float *)PyArray_DATA(arr1);
  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nr,nc;
  nr = dims[0];
  nc = dims[1];

  int i,j;
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      fprintf(fp,fmt,*(val++));
    }
    fprintf(fp,"\n");
  }

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_INCREF(Py_None);
  return Py_None;

 fail:
  Py_XDECREF(arr1);
  return NULL;
}

/**************************************************** tofile_int32 ****/
/* Write an array to file */

static PyObject * tofile_int32(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL;
  PyObject *arr1=NULL;
  FILE *fp;
  char *fmt; /* single element format */

  if (!PyArg_ParseTuple(args, "OOs", &arg1, &arg2, &fmt)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_INT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  fp = PyFile_AsFile(arg2);
  if (!fp) goto fail;

  int *val;
  val = (int *)PyArray_DATA(arr1);
  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nr,nc;
  nr = dims[0];
  nc = dims[1];

  int i,j;
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      fprintf(fp,fmt,*(val++));
    }
    fprintf(fp,"\n");
  }

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_INCREF(Py_None);
  return Py_None;

 fail:
  Py_XDECREF(arr1);
  return NULL;
}

/**************************************************** tofile_ifloat32 ****/
/* Write an indexed array to file */
/*
   Use:  tofile_ifloat32(ind,data,file,format)
   This is like tofile_float32, but each row from data is preceded with an 
   index number from ind.
*/
    

static PyObject * tofile_ifloat32(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
  PyObject *arr1=NULL, *arr2=NULL;
  FILE *fp;
  char *fmt; /* single element format */

  if (!PyArg_ParseTuple(args, "OOOs", &arg1, &arg2, &arg3, &fmt)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_INT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  fp = PyFile_AsFile(arg3);
  if (!fp) goto fail;

  float *val;
  val = (float *)PyArray_DATA(arr2);
  npy_intp * dims;
  dims = PyArray_DIMS(arg2);
  int nr,nc;
  nr = dims[0];
  nc = dims[1];

  int *ind;
  ind = (int *)PyArray_DATA(arr1);
  dims = PyArray_DIMS(arg1);
  if (dims[0] != nr) goto fail;

  int i,j;
  for (i=0; i<nr; i++) {
    fprintf(fp,"%i ",*(ind++));
    for (j=0; j<nc; j++) {
      fprintf(fp,fmt,*(val++));
    }
    fprintf(fp,"\n");
  }

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_INCREF(Py_None);
  return Py_None;

 fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}

/**************************************************** tofile_iint32 ****/
/* Write an indexed array to file */
/*
   Use:  tofile_iint32(ind,data,file,format)
   This is like tofile_int32, but each row from data is preceded with an 
   index number from ind.
*/
    

static PyObject * tofile_iint32(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
  PyObject *arr1=NULL, *arr2=NULL;
  FILE *fp;
  char *fmt; /* single element format */

  if (!PyArg_ParseTuple(args, "OOOs", &arg1, &arg2, &arg3, &fmt)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_INT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;
  fp = PyFile_AsFile(arg3);
  if (!fp) goto fail;

  int *val;
  val = (int *)PyArray_DATA(arr2);
  npy_intp * dims;
  dims = PyArray_DIMS(arg2);
  int nr,nc;
  nr = dims[0];
  nc = dims[1];

  int *ind;
  ind = (int *)PyArray_DATA(arr1);
  dims = PyArray_DIMS(arg1);
  if (dims[0] != nr) goto fail;

  int i,j;
  for (i=0; i<nr; i++) {
    fprintf(fp,"%i ",*(ind++));
    for (j=0; j<nc; j++) {
      fprintf(fp,fmt,*(val++));
    }
    fprintf(fp,"\n");
  }

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_INCREF(Py_None);
  return Py_None;

 fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}


/**************************************************** coords_fuse ****/
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
    out   : (nelems,nplex,nval) return values where each value is
    replaced with the sum of its value at that node.
    If avg=True, the values are replaced with the average instead.
    (CURRENTLY NOT IMPLEMENTED)
    The operations are can be done in-place by specifying the same array
    for val and out. 
*/  
  

static void nodal_sum(float *val, int *elems, float *out, int nelems, int nplex, int nval, int maxnod, int avg, int all)
{
  int i,k,n,nnod=maxnod+1;
  int *cnt = (int *) malloc(nnod*sizeof(int));
  float *work;

  if (all)
    work = (float *) malloc(nnod*nval*sizeof(float));
  else
    work = out;

  for (i=0; i<nnod; i++) cnt[i] = 0;
  for (i=0; i<nnod*nval; i++) work[i] = 0.0;

  /* Loop over the input and sum */
  for (i=0; i<nelems*nplex; i++) {
    n = elems[i];
    for (k=0; k<nval; k++) work[n*nval+k] += val[i*nval+k];
    cnt[n]++;
  }
  /* Divide by count */
  if (avg)
    for (i=0; i<=maxnod; i++)
      if (cnt[n] > 0)
  	for (k=0; k<nval; k++) work[n*nval+k] /= cnt[n];
  
  /* Place back results */
  if (all)
    for (i=0; i<nelems*nplex; i++) {
      n = elems[i];
      for (k=0; k<nval; k++) out[i*nval+k] = work[n*nval+k];
  }

  if (all)
    free(work);
  free(cnt);
}


static PyObject * nodalSum(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL, *ret=NULL;
  PyObject *arr1=NULL, *arr2=NULL;
  float *val,*out;
  int *elems;
  int avg, all, max;
  if (!PyArg_ParseTuple(args, "OOiii", &arg1, &arg2, &max, &avg, &all)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;

  npy_intp * dim;
  int nelems,nplex,nval;
  dim = PyArray_DIMS(arg1);
  nelems = dim[0];
  nplex = dim[1];
  nval = dim[2];

  val = (float *)PyArray_DATA(arr1);
  elems = (int *)PyArray_DATA(arr2);
 
  /* create return  array */
  if (all)
    ret = PyArray_SimpleNew(3,dim, NPY_FLOAT);
  else {
    dim[0] = max+1;
    dim[1] = nval;
    ret = PyArray_SimpleNew(2,dim, NPY_FLOAT);
  }
  out = (float *)PyArray_DATA(ret);

  /* compute */
  nodal_sum(val,elems,out,nelems,nplex,nval,max,avg,all);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  return ret;
 fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}


/**************************************************** average_direction ****/
/* Average of vectors within tolerance */
/* args:  vec, tol
    vec   : (nvec,ndim) normalized vectors.
    tol   : tolerance on direction

    The vectors are supposed to be normalized.
    The operations are done in-place. The return value is None.
*/  

static void average_direction(float *vec, int nvec, int ndim, float tol)
{
  int i,j,k, cnt, *par;
  float p;
  
  par = (int *) malloc(nvec*sizeof(int));

  for (i=0; i<nvec; i++) par[i] = -1;
  j = 0;
  //printf("nvec=%d, ndim=%d\n",nvec,ndim);
  while (j < nvec) {
    par[j] = j;
    /* mark the close directions */
    for (i=j+1; i<nvec; i++) {
      p = dotprod(vec+j*ndim,1,vec+i*ndim,1,ndim);
      //printf("Proj = %f, %f\n",p,tol);
      if (p >= tol) {
	//printf("setting %d = %d\n",i,j);
	par[i] = j;
      }
    }
    /* average the close directions */
    cnt = 1;
    for (i=j+1; i<nvec; i++) {
      if (par[i] == j) {
	cnt++;
	for (k=0; k<ndim; k++) vec[j*ndim+k] += vec[i*ndim+k];
      }
    }
    for (k=0; k<ndim; k++) vec[j*ndim+k] /= cnt;
    /* check if untreated vectors left */
    //for (i=0; i<nvec; i++) printf("par[%d] = %d\n",i,par[i]);
    for (i=j+1; i<nvec; i++)
      if (par[i] < 0)
	break;
    j = i;
    //printf("Continuing from %d\n",j);
  }
  /* copy average vectors to other positions */
  //for (i=0; i<nvec; i++) printf("par[%d] = %d\n",i,par[i]);
  for (i=0; i<nvec; i++) {
    j = par[i];
    if (j < i) {
      //printf("Copying %d to %d\n",j,i);
      for (k=0; k<ndim; k++) vec[i*ndim+k] = vec[j*ndim+k];
    }
  }
  free(par);
}

static void average_direction_indexed(float *vec, int ndim, int*ind, int nvec, float tol)
{
  int i,j,k, cnt, *par;
  float p;
  
  par = (int *) malloc(nvec*sizeof(int));

  for (i=0; i<nvec; i++) par[i] = -1;
  j = 0;
  //printf("nvec=%d, ndim=%d\n",nvec,ndim);
  while (j < nvec) {
    par[j] = j;
    /* mark the close directions */
    for (i=j+1; i<nvec; i++) {
      p = dotprod(vec+ind[j]*ndim,1,vec+ind[i]*ndim,1,ndim);
      //printf("Proj = %f, %f\n",p,tol);
      if (p >= tol) {
	//printf("setting %d = %d\n",i,j);
	par[i] = j;
      }
    }
    /* average the close directions */
    cnt = 1;
    for (i=j+1; i<nvec; i++) {
      if (par[i] == j) {
	cnt++;
	for (k=0; k<ndim; k++) vec[ind[j]*ndim+k] += vec[ind[i]*ndim+k];
      }
    }
    for (k=0; k<ndim; k++) vec[ind[j]*ndim+k] /= cnt;
    /* check if untreated vectors left */
    //for (i=0; i<nvec; i++) printf("par[%d] = %d\n",i,par[i]);
    for (i=j+1; i<nvec; i++)
      if (par[i] < 0)
	break;
    j = i;
    //printf("Continuing from %d\n",j);
  }
  /* copy average vectors to other positions */
  //for (i=0; i<nvec; i++) printf("par[%d] = %d\n",i,par[i]);
  for (i=0; i<nvec; i++) {
    j = par[i];
    if (j < i) {
      //printf("Copying %d to %d\n",j,i);
      for (k=0; k<ndim; k++) vec[ind[i]*ndim+k] = vec[ind[j]*ndim+k];
    }
  }
  free(par);
}




static PyObject * averageDirection(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL;
  PyObject *arr1=NULL;
  float *vec, tol;
  if (!PyArg_ParseTuple(args, "Of", &arg1, &tol)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr1 == NULL) return NULL;

  npy_intp * dims;
  dims = PyArray_DIMS(arg1);
  int nvec,ndim;
  nvec = dims[0];
  ndim = dims[1];

  vec = (float *)PyArray_DATA(arr1);
  average_direction(vec,nvec,ndim,tol);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * averageDirectionIndexed(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL;
  PyObject *arr1=NULL, *arr2=NULL;
  int *ind;
  float *vec, tol;
  if (!PyArg_ParseTuple(args, "OOf", &arg1, &arg2, &tol)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_INT, NPY_INOUT_ARRAY);
  if (arr2 == NULL) goto fail;

  npy_intp * dims;
  int nvec,ndim;
  dims = PyArray_DIMS(arg1);
  ndim = dims[1];
  dims = PyArray_DIMS(arg2);
  nvec = dims[0];

  vec = (float *)PyArray_DATA(arr1);
  ind = (int *)PyArray_DATA(arr2);
  average_direction_indexed(vec,ndim,ind,nvec,tol);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}


/********************************************************/
/* The methods defined in this module */
static PyMethodDef _methods_[] = {
    {"_fuse", coords_fuse, METH_VARARGS, "Fuse nodes."},
    {"nodalSum", nodalSum, METH_VARARGS, "Nodal sum."},
    {"averageDirection", averageDirection, METH_VARARGS, "Average directions."},
    {"averageDirectionIndexed", averageDirectionIndexed, METH_VARARGS, "Average directions."},
    {"tofile_float32", tofile_float32, METH_VARARGS, "Write float32 array to file."},
    {"tofile_int32", tofile_int32, METH_VARARGS, "Write int32 array to file."},
    {"tofile_ifloat32", tofile_ifloat32, METH_VARARGS, "Write indexed float32 array to file."},
    {"tofile_iint32", tofile_iint32, METH_VARARGS, "Write indexed int32 array to file."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module */
PyMODINIT_FUNC initmisc_(void)
{
  PyObject* module;
  module = Py_InitModule3("misc_", _methods_, __doc__);
  PyModule_AddIntConstant(module,"accelerated",1);
  import_array(); /* Get access to numpy array API */
}

/* End */
