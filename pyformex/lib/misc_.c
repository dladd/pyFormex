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
#include <math.h>

static char __doc__[] = "misc_ module\n\
\n\
This module provides accelerated versions of miscellaneous pyFormex functions.\n\
\n";

/* Dot product of two vectors of length n */
/* ia and ib are the strides of the elements addressed starting from a, b */ 
float dotprod(float *a, int ia, float *b, int ib, int n)
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
  

void nodal_sum(float *val, int *elems, float *out, int nelems, int nplex, int nval, int maxnod, int avg, int all)
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
    for (n=0; n<=maxnod; n++)
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

void average_direction(float *vec, int nvec, int ndim, float tol)
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

void average_direction_indexed(float *vec, int ndim, int*ind, int nvec, float tol)
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


/**************************************************** isosurface ****/
/* Create an isosurface through data at given level */
/* args: data, level
   data  : (nx,ny,nz) shaped array of data values at points with
           coordinates equal to their indices. This defines a 3D volume
           [0,nx-1], [0,ny-1], [0,nz-1]
   level : data value at which the isosurface is to be constructed

   Returns an (ntr,3,3) array defining the triangles of the isosurface.
   The result may be empty (if level is outside the data range).

   This Code is based on the example by Paul Bourke from
   http://paulbourke.net/geometry/polygonise/
*/
#define FLOAT float
#define ABS fabs

typedef struct {
  FLOAT x;
  FLOAT y;
  FLOAT z;
} XYZ;

typedef union {
  XYZ p;
  FLOAT x[3];
} POINT;
  

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value

   p1,p2: coordinates of the two vertices
   v1,v2: values at the vertices
   level: isosurface level
*/

XYZ VertexInterp(XYZ p1, XYZ p2, FLOAT v1, FLOAT v2, FLOAT level)
{
  FLOAT mu;
  XYZ p;
  
  if (ABS(level-v1) < 0.00001) return(p1);
  if (ABS(level-v2) < 0.00001) return(p2);
  if (ABS(v1-v2) < 0.00001)    return(p1);
  mu = (level - v1) / (v2 - v1);
  p.x = p1.x + mu * (p2.x - p1.x);
  p.y = p1.y + mu * (p2.y - p1.y);
  p.z = p1.z + mu * (p2.z - p1.z);
  return(p);
}

/*
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
   0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
*/
int Polygonise(FLOAT *triangles, XYZ *pos, FLOAT *val, FLOAT level)
{
  int edgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
  
  int triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
  
  /* Edge connections: 2 vertices per edge*/
  int edge_con[12][2] = {
    {0,1},
    {1,2},
    {2,3},
    {3,0},
    {4,5},
    {5,6},
    {6,7},
    {7,4},
    {0,4},
    {1,5},
    {2,6},
    {3,7}};

  int i,j,k,ntriang;
  int cubeindex;
  POINT vertlist[12];
  
  /*
    Determine the index into the edge table which
    tells us which vertices are inside of the surface
  */
  cubeindex = 0;
  for (i=0; i<8; i++)
    if (val[i] < level)
      cubeindex |= 1 << i;
  
  /* Cube is entirely in/out of the surface */
  if (edgeTable[cubeindex] == 0)
    return(0);
  
  /* Find the vertices where the surface intersects the cube */
  for (i=0; i<12; i++)
    if (edgeTable[cubeindex] & (1 << i)) {
      j = edge_con[i][0];
      k = edge_con[i][1];
      vertlist[i].p = VertexInterp(pos[j],pos[k],val[j],val[k],level);
    }

   /* Create the triangles */
   ntriang = 0;
   for (i=0; triTable[cubeindex][i]!=-1; i+=3) {
     for (j=0; j<3; j++)   /* loop over vertices */
       for (k=0; k<3; k++)  /* loop over coordinates */
	 *triangles++ = vertlist[triTable[cubeindex][i+j]].x[k];
     ntriang++;
   }

   return(ntriang);
}


static char isosurface__doc__[] = "Create an isosurface through data at given level.\n\
\n\
    - `data`: (nx,ny,nz) shaped array of data values at points with\n\
      coordinates equal to their indices. This defines a 3D volume\n\
      [0,nx-1], [0,ny-1], [0,nz-1]\n\
    - `level`: data value at which the isosurface is to be constructed\n\
\n\
    Returns an (ntr,3,3) array defining the triangles of the isosurface.\n\
    The result may be empty (if level is outside the data range).\n\
\n\
    The his function uses the marching cube algorithm for the reconstruction.\n\
    See http://paulbourke.net/geometry/polygonise/\n\
";

static PyObject * isosurface(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL;
  PyObject *arr1=NULL;
  float *data;
  float level;
  if (!PyArg_ParseTuple(args, "Of", &arg1, &level)) return NULL;
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr1 == NULL) return NULL;

  npy_intp * dims;
  int nx,ny,nz;
  dims = PyArray_DIMS(arg1);
  nz = dims[0];
  ny = dims[1];
  nx = dims[2];
  data = (float *)PyArray_DATA(arr1);
 
  /* allocate memory for triangles */
  int nitri = 2*nx*ny*nz; /* initial guess for number of triangles */
  int ntri = 0;   /* size of storage available */ 
  int itri = 0;   /* size of storage filled */
  float *triangles = NULL; /* pointer to storage size */

  /* create array to store triangles for one cell (max 5) */
  /* TRIANGLE cubetri[5]; */

  /* vertex coordinates with respect to ix,iy,iz */
  int grid[8][3] = {
    {0,0,0},
    {1,0,0},
    {1,1,0},
    {0,1,0},
    {0,0,1},
    {1,0,1},
    {1,1,1},
    {0,1,1},
  };

  /* data offsets with respect to first vertex data */
  int i,ofs[8];
  //printf("Data offsets\n");
  for (i=0; i<8; i++) { 
    ofs[i] = ( grid[i][2]*ny + grid[i][1] )*nx + grid[i][0];
    //printf("%d\n",ofs[i]);
  }

  /* loop over cells */
  int mtri;
  int ix,iy,iz;
  XYZ pos[8];   /* coordinates of the cell vertices */
  FLOAT val[8]; /* data values at the cell vertices */
  int iofs;     /* data offset of vertex ix,iy,iz */
  for (iz=0; iz<nz-1; iz++) {
    for (i=0; i<8; i++) pos[i].z = iz + grid[i][2];
    for (iy=0; iy<ny-1; iy++) {
      for (i=0; i<8; i++) pos[i].y = iy + grid[i][1];
      for (ix=0; ix<nx-1; ix++) {
	for (i=0; i<8; i++) pos[i].x = ix + grid[i][0];
	iofs = (iz*ny + iy)*nx + ix;
	for (i=0; i<8; i++) val[i] = data[iofs + ofs[i]];
	if (itri+5 > ntri) {
	  /* need to enlarge storage */
	  ntri += nitri;
	  triangles = (float*) realloc(triangles,ntri*3*3*sizeof(float));
	}
	//printf("Values\n");
	for (i=0; i<8; i++) { 
	  //printf("%f, %f, %f : %f\n",pos[i].x,pos[i].y,pos[i].z,val[i]);
	}
	mtri = Polygonise(triangles+itri*3*3,pos,val,level);
	itri += mtri;
	//printf("%d new triangles; %d total triangles; %d max triangles\n",mtri,itri,ntri);
      }
    }
  }

  /* create return array */
  npy_intp dim[3];
  dim[0] = itri;
  dim[1] = 3;
  dim[2] = 3;
  PyObject *ret = PyArray_SimpleNew(3,dim, NPY_FLOAT);
  float *out = (float *)PyArray_DATA(ret);
  memcpy(out,triangles,itri*3*3*sizeof(float));

  /* Clean up and return */
  Py_DECREF(arr1);
  return ret;
}


/********************************************************/
/* The methods defined in this module */
static PyMethodDef _methods_[] = {
    {"_fuse", coords_fuse, METH_VARARGS, "Fuse nodes."},
    {"nodalSum", nodalSum, METH_VARARGS, "Nodal sum."},
    {"averageDirection", averageDirection, METH_VARARGS, "Average directions."},
    {"averageDirectionIndexed", averageDirectionIndexed, METH_VARARGS, "Average directions."},
    {"isosurface", isosurface, METH_VARARGS, isosurface__doc__},
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
