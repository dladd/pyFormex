/* $Id */
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

//
// This is based on the Nurbs toolbox Python port by 
// Runar Tenfjord (http://www.aria.uklinux.net/nurbs.php3)
//
// It was adapted to numpy by Benedict Verhegghe
//

#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>


static char _nurbs_module__doc__[] = "_nurbs_ module. Version 0.1\n\
\n\
This module implements low level NURBS functions.\n\
\n";

int min(int a, int b)
{
  if (b < a) a = b;
  return a;
}

int max(int a, int b)
{
  if (b > a) a = b;
  return a;
}


static double **vec2mat(double *vec, int nrows, int ncols) 
{
  int row;
  double **mat;

  mat = (double**) malloc (nrows*sizeof(double*));
  mat[0] = vec;
  for (row = 1; row < nrows; row++)
    mat[row] = mat[row-1] + ncols;  
  return mat;
}

static double **matrix(int nrows, int ncols) 
{
  int row;
  double **mat;

  mat = (double**) malloc (nrows*sizeof(double*));
  mat[0] = (double*) malloc (nrows*ncols*sizeof(double));
  for (row = 1; row < nrows; row++)
    mat[row] = mat[row-1] + ncols;  
  return mat;
}

static void freematrix(double **mat)
{
  free(mat[0]);
  free(mat);
}

// Compute logarithm of the gamma function
// Algorithm from 'Numerical Recipes in C, 2nd Edition' pg214.
static double _gammaln(double xx)
{
  double x,y,tmp,ser;
  static double cof[6] = {76.18009172947146,-86.50532032291677,
                          24.01409824083091,-1.231739572450155,
                          0.12086650973866179e-2, -0.5395239384953e-5};
  int j;
  y = x = xx;
  tmp = x + 5.5;
  tmp -= (x+0.5) * log(tmp);
  ser = 1.000000000190015;
  for (j=0; j<=5; j++) ser += cof[j]/++y;
  return -tmp+log(2.5066282746310005*ser/x);
}

// computes ln(n!)
// Numerical Recipes in C
// Algorithm from 'Numerical Recipes in C, 2nd Edition' pg215.
static double _factln(int n)
{
  static int ntop = 0;
  static double a[101];
  
  if (n <= 1) return 0.0;
  while (n > ntop)
  {
    ++ntop;
    a[ntop] = _gammaln(ntop+1.0);
  }
  return a[n];
}

static char bincoeff__doc__[] =
"Computes the binomial coefficient.\n\
\n\
 ( n )      n!\n\
 (   ) = --------\n\
 ( k )   k!(n-k)!\n\
\n\
 Algorithm from 'Numerical Recipes in C, 2nd Edition' pg215.\n";

static double _bincoeff(int n, int k)
{
  return floor(0.5+exp(_factln(n)-_factln(k)-_factln(n-k)));
}

static PyObject * _nurbs_bincoeff(PyObject *self, PyObject *args)
{
  int n, k;
  double ret;
  if(!PyArg_ParseTuple(args, "ii", &n, &k))
    return NULL;
  ret = _bincoeff(n, k);
  return Py_BuildValue("d",ret);
}

// Find the knot span of the parametric point u. 
//
// INPUT:
//
//   n - number of control points - 1
//   p - spline degree       
//   u - parametric point    
//   U - knot sequence
//
// RETURN:
//
//   s - knot span
//
// Algorithm A2.1 from 'The NURBS BOOK' pg68.
static int _findspan(int n, int p, double u, double *U, int nU)
{
  int low, high, mid;
  int cnt=0;
  printf("findspan %d %d %f\n",n,p,u);

  // special case
  if (u == U[n+1]) return(n);
    
  // do binary search
  //low = p;
  //high = n + 1;
  // BV !!!
  low = 0;
  high = nU-1;
  mid = (low + high) / 2;
  printf("low = %d, high = %d, mid = %d\n",low,high,mid);
  while (u < U[mid] || u > U[mid+1])
  {
    printf("mid = %d\n",mid);
    printf("%f < %f < %f\n",U[mid],u,U[mid+1]);
    if (u < U[mid])
      high = mid;
    else
      low = mid;
    mid = (low + high) / 2;
    cnt ++;
    if (cnt > 20) break;
  }  

  return(mid);
}

// Basis Function. 
//
// INPUT:
//
//   i - knot span  ( from FindSpan() )
//   u - parametric point
//   p - spline degree
//   U - knot sequence
//
// OUTPUT:
//
//   N - Basis functions vector[p+1]
//
// Algorithm A2.2 from 'The NURBS BOOK' pg70.

static void _basisfuns(int i, double u, int p, double *U, double *N)
{
  int j,r;
  double saved, temp;

  // work space
  double *left  = (double*) malloc((p+1)*sizeof(double));
  double *right = (double*) malloc((p+1)*sizeof(double));
  
  N[0] = 1.0;
  for (j = 1; j <= p; j++)
  {
    left[j]  = u - U[i+1-j];
    right[j] = U[i+j] - u;
    saved = 0.0;
    
    for (r = 0; r < j; r++)
    {
      temp = N[r] / (right[r+1] + left[j-r]);
      N[r] = saved + right[r+1] * temp;
      saved = left[j-r] * temp;
    } 

    N[j] = saved;
  }
  
  free(left);
  free(right);
}

static char bspeval__doc__[] =
"Evaluation of univariate B-Spline. \n\
\n\
INPUT:\n\
\n\
 d - spline degree       integer\n\
 c - control points      double  matrix(mc,nc)\n\
 k - knot sequence       double  vector(nk)\n\
 u - parametric points   double  vector(nu)\n\
\n\
OUTPUT:\n\
\n\
   p - evaluated points    double  matrix(mc,nu)\n\
\n\
Modified version of Algorithm A3.1 from 'The NURBS BOOK' pg82.\n\
\n";

static void _bspeval(int d, double **ctrl, int mc, int nc, double *k, int nk, double *u, int nu, double **pnt)
{
  int i, s, tmp1, row, col;
  double tmp2;
  
  printf("This is the evaluator\n");
  // space for the basis functions
  printf("Allocating space\n");
  double *N = (double*) malloc((d+1)*sizeof(double));

  for (col = 0; col < nc; col++) printf("%f %f %f\n",ctrl[0][col],ctrl[1][col],ctrl[2][col]);

  // for each parametric point i
  for (col = 0; col < nu; col++) {
    printf("point %d = %f\n",col,u[col]);
    // find the span of u[col]
    s = _findspan(nc-1, d, u[col], k,nk);
    printf("span %d\n",s);
    _basisfuns(s, u[col], d, k, N);
    for (i = 0; i <= d; i++) printf("basis %d = %f\n",i,N[i]);
    for (row = 0; row < mc; row++) pnt[row][col] = 0.0;

    tmp1 = s - d;
    for (row = 0; row < mc; row++) {
      tmp2 = 0.0;   
      for (i = 0; i <= d; i++) tmp2 += N[i] * ctrl[row][tmp1+i];
      pnt[row][col] = tmp2;
    }
  }
  free(N);
} 

static PyObject * _nurbs_bspeval(PyObject *self, PyObject *args)
{
  int d, mc, nc, nk, nu;
  npy_intp *ctrl_dim, *k_dim, *u_dim, dim[2];
  double *ctrl, *k, *u, *pnt;
  double **ctrlmat, **pntmat;
  PyObject *arg2, *arg3, *arg4;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret=NULL;

  if (!PyArg_ParseTuple(args, "iOOO", &d, &arg2, &arg3, &arg4))
    return NULL;
  arr1 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr1 == NULL)
    return NULL;
  arr2 = PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr2 == NULL)
    goto fail;
  arr3 = PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr3 == NULL)
    goto fail;

  /* We suppose the dimensions are correct*/
  ctrl_dim = PyArray_DIMS(arr1);
  k_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  mc = ctrl_dim[0];
  nc = ctrl_dim[1];
  nk = k_dim[0];
  nu = u_dim[0];
  ctrl = (double *)PyArray_DATA(arr1);
  k = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);
  dim[0] = mc;
  dim[1] = nu;
  printf("%d %d %d %d %d\n",d,mc,nc,nk,nu);

  /* Create the return array */
  ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  printf("got new array\n");
  pnt = (double *)PyArray_DATA(ret);

  ctrlmat = vec2mat(ctrl, mc, nc);
  pntmat = vec2mat(pnt, mc, nu);
  printf("converted to matrices\n");
  _bspeval(d, ctrlmat, mc, nc, k, nk, u, nu, pntmat);
  printf("return from computations\n");
  free(ctrlmat);
  free(pntmat);

  /* Clean up and return */
  printf("cleanup and return\n");
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  return ret;

 fail:
  printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  return NULL;
}

// Compute Non-zero basis functions and their derivatives.
//
// INPUT:
//
//   d  - spline degree         integer
//   k  - knot sequence         double  vector(nk)
//   u  - parametric point      double
//   s  - knot span             integer
//   n  - number of derivatives integer
//
// OUTPUT:
//
//   dN -  Basis functions      double  matrix(n+1,d+1)
//         and derivatives upto the nth derivative (n < d)
//
// Algorithm A2.3 from 'The NURBS BOOK' pg72.
static void _dersbasisfuns(int d, double *k, int nk, double u, int s,int n, double **ders)
{
  int i,j,r,s1,s2,rk,pk,j1,j2;
  double temp, saved, der;
  double **ndu, **a, *left, *right;

  ndu = matrix(d+1, d+1);
  a = matrix(d+1, 2);
  left = (double *) malloc((d+1)*sizeof(double));
  right = (double *) malloc((d+1)*sizeof(double));

  ndu[0][0] = 1.0;
  
  for( j = 1; j <= d; j++ )
  {
    left[j] = u - k[s+1-j];
    right[j] = k[s+j]-u;
    saved = 0.0;
    for( r = 0; r < j; r++ )
    {
      ndu[r][j] = right[r+1] + left[j-r];
      temp = ndu[j-1][r]/ndu[r][j];
      
      ndu[j][r] = saved + right[r+1]*temp;
      saved = left[j-r]*temp;
    }
    ndu[j][j] = saved;
  }

  for( j = 0; j <= d; j++ )
    ders[j][0] = ndu[d][j];

  for( r = 0; r <= d; r++ )
  {
    s1 = 0;    s2 = 1;
    a[0][0] = 1.0;

    for( i = 1; i <= n; i++ )
    {
      der = 0.0;
      rk = r-i;  pk = d-i;
      
      if( r >= i )
      {
        a[0][s2] = a[0][s1] / ndu[rk][pk+1];
        der = a[0][s2] * ndu[pk][rk];
      }  
      if( rk >= -1 )
        j1 = 1;
      else
        j1 = -rk;  
      if( r-1 <= pk )
        j2 = i-1;
      else
        j2 = der-r;  

      for( j = j1; j <= j2; j++ )
      {
        a[j][s2] = (a[j][s1] - a[j-1][s1]) / ndu[rk+j][pk+1];
        der += a[j][s2] * ndu[pk][rk+j];
      }  
      if( r <= pk )
      {
        a[i][s2] = -a[i-1][s1] / ndu[r][pk+1];
        der += a[i][s2] * ndu[pk][r];
      }  
      ders[r][i] = der;
      j = s1; s1 = s2; s2 = j;
    }        
  }
  
  r = d;
  for( i = 1; i <= n; i++ )
  {
    for( j = 0; j <= d; j++ )
      ders[j][i] *= r;
    r *= d-i;
  }    

  freematrix(ndu);
  freematrix(a);
  free(left);
  free(right);
}

static char bspdeval__doc__[] =
"Evaluate a B-Spline derivative curve.\n\
\n\
INPUT:\n\
\n\
 d - spline degree       integer\n\
 c - control points      double  matrix(mc,nc)\n\
 k - knot sequence       double  vector(nk)\n\
 u - parametric point    double\n\
 n - nth derivative      integer\n\
\n\
OUTPUT:\n\
\n\
 p - evaluated points    double  matrix(mc, n+1)\n\
\n\
Modified version of Algorithm A3.2 from 'The NURBS BOOK' pg93.\n\
\n";

static void _bspdeval(int d, double **c, int mc, int nc, double *k, int nk, 
             double u, int n, double **p)
{
  int i, l, j, s;
  int du = min(d,n);
  double **dN;   

  dN = matrix(d+1, n+1);

  for (l = d+1; l <= n; l++)
    for (i = 0; i < mc; i++)
      p[l][i] = 0.0;

  s = _findspan(nc-1, d, u, k,nk);
  _dersbasisfuns(d, k, nk, u, s, n, dN);

  for (l = 0; l <= du; l++)
  {
    for (i = 0; i < mc; i++)
    {
      p[l][i] = 0.0;
      for (j = 0; j <= d; j++)
        p[l][i] += dN[l][j] * c[s-d+j][i];
    }
  }
  freematrix(dN); 
}

static PyObject * _nurbs_bspdeval(PyObject *self, PyObject *args)
{
  int d, mc, nc, n, dim[2];
  double u, **ctrlmat, **pntmat;
  PyObject *arg2, *arg3;
  PyArrayObject *ctrl, *k, *pnt;
  if(!PyArg_ParseTuple(args, "iOOdi", &d, &arg2, &arg3, &u, &n))
    return NULL;
  ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2);
  if(ctrl == NULL)
    return NULL;
  k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1);
  if(k == NULL)
    return NULL;
  mc = ctrl->dimensions[0];
  nc = ctrl->dimensions[1];
  dim[0] = mc;
  dim[1] = n + 1;
  pnt = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE);
  ctrlmat = vec2mat(ctrl->data, mc, nc);
  pntmat = vec2mat(pnt->data, mc, n + 1);
  _bspdeval(d, ctrlmat, mc, nc, (double *)k->data, k->dimensions[0], u, n, pntmat);
  free(pntmat);
  free(ctrlmat);
  Py_DECREF(ctrl);
  Py_DECREF(k);
  return PyArray_Return(pnt);
}

static char bspkntins__doc__[] =
"Insert Knot into a B-Spline.\n\
\n\
INPUT:\n\
\n\
 d - spline degree       integer\n\
 c - control points      double  matrix(mc,nc)\n\
 k - knot sequence       double  vector(nk)\n\
 u - new knots           double  vector(nu)\n\
\n\
OUTPUT:\n\
\n\
 ic - new control points double  matrix(mc,nc+nu)\n\
 ik - new knot sequence  double  vector(nk+nu)\n\
\n\
Modified version of Algorithm A5.4 from 'The NURBS BOOK' pg164.\n\
\n";

static void _bspkntins(int d, double **ctrl, int mc, int nc, double *k, int nk, 
              double *u, int nu, double **ictrl, double *ik)
{
  int a, b, r, l, i, j, m, n, s, q, ind;
  double alfa;

  n = nc - 1;
  r = nu - 1;

  m = n + d + 1;
  a = _findspan(n, d, u[0], k,nk);
  b = _findspan(n, d, u[r], k,nk);
  ++b;

  for (q = 0; q < mc; q++)
  {
    for (j = 0; j <= a-d; j++) ictrl[q][j] = ctrl[q][j];
    for (j = b-1; j <= n; j++) ictrl[q][j+r+1] = ctrl[q][j];
  }
  for (j = 0; j <= a; j++)   ik[j] = k[j];
  for (j = b+d; j <= m; j++) ik[j+r+1] = k[j];

  i = b + d - 1;
  s = b + d + r;
  for (j = r; j >= 0; j--)
  {
    while (u[j] <= k[i] && i > a)
    {
      for (q = 0; q < mc; q++)
        ictrl[q][s-d-1] = ctrl[q][i-d-1];
      ik[s] = k[i];
      --s;
      --i;
    }
    for (q = 0; q < mc; q++)
      ictrl[q][s-d-1] = ictrl[q][s-d];
    for (l = 1; l <= d; l++)
    {
      ind = s - d + l;
      alfa = ik[s+l] - u[j];
      if (fabs(alfa) == 0.0)
        for (q = 0; q < mc; q++)
          ictrl[q][ind-1] = ictrl[q][ind];
      else
      {
        alfa /= (ik[s+l] - k[i-d+l]);
        for (q = 0; q < mc; q++)
          ictrl[q][ind-1] = alfa*ictrl[q][ind-1]+(1.0-alfa)*ictrl[q][ind];
      }
    }

    ik[s] = u[j];
    --s;
  }
}

static PyObject * _nurbs_bspkntins(PyObject *self, PyObject *args)
{
  int d, mc, nc, nk, nu, dim[2];
  double **ctrlmat, **icmat;
  PyObject *arg2, *arg3, *arg4;
  PyArrayObject *ctrl, *k, *u, *ic, *ik;
  if(!PyArg_ParseTuple(args, "iOOO", &d, &arg2, &arg3, &arg4))
    return NULL;
  ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2);
  if(ctrl == NULL)
    return NULL;
  k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1);
  if(k == NULL)
    return NULL;
  u = (PyArrayObject *) PyArray_ContiguousFromObject(arg4, PyArray_DOUBLE, 1, 1);
  if(u == NULL)
    return NULL;
  mc = ctrl->dimensions[0];
  nc = ctrl->dimensions[1];
  nk = k->dimensions[0];
  nu = u->dimensions[0];
  dim[0] = mc;
  dim[1] = nc + nu;
  ic = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE);
  ctrlmat = vec2mat(ctrl->data, mc, nc);
  icmat = vec2mat(ic->data, mc, nc + nu);
  dim[0] = nk + nu;
  ik = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  _bspkntins(d, ctrlmat, mc, nc, (double *)k->data, nk, (double *)u->data, nu, icmat, (double *)ik->data);
  free(icmat);
  free(ctrlmat);
  Py_DECREF(ctrl);
  Py_DECREF(k);
  Py_DECREF(u);
  return Py_BuildValue("(OO)", (PyObject *)ic, (PyObject *)ik);
}

static char bspdegelev__doc__[] =
"Degree elevate a B-Spline t times.\n\
\n\
INPUT:\n\
\n\
 n,p,U,Pw,t\n\
\n\
OUTPUT:\n\
\n\
 nh,Uh,Qw\n\
\n\
Modified version of Algorithm A5.9 from 'The NURBS BOOK' pg206.\n\
\n";

static void _bspdegelev(int d, double **ctrl, int mc, int nc, double *k, int nk, 
               int t, int *nh, double **ictrl, double *ik)
{
  int i, j, q, s, m, ph, ph2, mpi, mh, r, a, b, cind, oldr, mul;
  int n, lbz, rbz, save, tr, kj, first, kind, last, bet, ii;
  double inv, ua, ub, numer, den, alf, gam;
  double **bezalfs, **bpts, **ebpts, **Nextbpts, *alfs; 

  n = nc - 1;

  bezalfs = matrix(d+1,d+t+1);
  bpts = matrix(mc,d+1);
  ebpts = matrix(mc,d+t+1);
  Nextbpts = matrix(mc,d);
  alfs = (double *) malloc(d*sizeof(double));

  m = n + d + 1;
  ph = d + t;
  ph2 = ph / 2;

  // compute bezier degree elevation coefficeients  
  bezalfs[0][0] = bezalfs[d][ph] = 1.0;

  for (i = 1; i <= ph2; i++)
  {
    inv = 1.0 / _bincoeff(ph,i);
    mpi = min(d,i);
    
    for (j = max(0,i-t); j <= mpi; j++)
      bezalfs[j][i] = inv * _bincoeff(d,j) * _bincoeff(t,i-j);
  }    
  
  for (i = ph2+1; i <= ph-1; i++)
  {
    mpi = min(d, i);
    for (j = max(0,i-t); j <= mpi; j++)
      bezalfs[j][i] = bezalfs[d-j][ph-i];
  }       

  mh = ph;
  kind = ph+1;
  r = -1;
  a = d;
  b = d+1;
  cind = 1;
  ua = k[0];
  for (ii = 0; ii < mc; ii++)
    ictrl[ii][0] = ctrl[ii][0];
  
  for (i = 0; i <= ph; i++)
    ik[i] = ua;
    
  // initialise first bezier seg
  for (i = 0; i <= d; i++)
    for (ii = 0; ii < mc; ii++)
      bpts[ii][i] = ctrl[ii][i];  

  // big loop thru knot vector
  while (b < m)
  {
    i = b;
    while (b < m && k[b] == k[b+1])
      b++;

    mul = b - i + 1;
    mh += mul + t;
    ub = k[b];
    oldr = r;
    r = d - mul;
    
    // insert knot u(b) r times
    if (oldr > 0)
      lbz = (oldr+2) / 2;
    else
      lbz = 1;

    if (r > 0)
      rbz = ph - (r+1)/2;
    else
      rbz = ph;  

    if (r > 0)
    {
      // insert knot to get bezier segment
      numer = ub - ua;
      for (q = d; q > mul; q--)
        alfs[q-mul-1] = numer / (k[a+q]-ua);
      for (j = 1; j <= r; j++)  
      {
        save = r - j;
        s = mul + j;            

        for (q = d; q >= s; q--)
          for (ii = 0; ii < mc; ii++)
            bpts[ii][q] = alfs[q-s]*bpts[ii][q]+(1.0-alfs[q-s])*bpts[ii][q-1];

        for (ii = 0; ii < mc; ii++)
          Nextbpts[ii][save] = bpts[ii][d];
      }  
    }
    // end of insert knot

    // degree elevate bezier
    for (i = lbz; i <= ph; i++)
    {
      for (ii = 0; ii < mc; ii++)
        ebpts[ii][i] = 0.0;
      mpi = min(d, i);
      for (j = max(0,i-t); j <= mpi; j++)
        for (ii = 0; ii < mc; ii++)
          ebpts[ii][i] = ebpts[ii][i] + bezalfs[j][i]*bpts[ii][j];
    }
    // end of degree elevating bezier

    if (oldr > 1)
    {
      // must remove knot u=k[a] oldr times
      first = kind - 2;
      last = kind;
      den = ub - ua;
      bet = (ub-ik[kind-1]) / den;
      
      // knot removal loop
      for (tr = 1; tr < oldr; tr++)
      {        
        i = first;
        j = last;
        kj = j - kind + 1;
        while (j - i > tr)
        {
          // loop and compute the new control points
          // for one removal step
          if (i < cind)
          {
            alf = (ub-ik[i])/(ua-ik[i]);
            for (ii = 0; ii < mc; ii++)
              ictrl[ii][i] = alf * ictrl[ii][i] + (1.0-alf) * ictrl[ii][i-1];
          }        
          if (j >= lbz)
          {
            if (j-tr <= kind-ph+oldr)
            {  
              gam = (ub-ik[j-tr]) / den;
              for (ii = 0; ii < mc; ii++)
                ebpts[ii][kj] = gam*ebpts[ii][kj] + (1.0-gam)*ebpts[ii][kj+1];
            }
            else
            {
              for (ii = 0; ii < mc; ii++)
                ebpts[ii][kj] = bet*ebpts[ii][kj] + (1.0-bet)*ebpts[ii][kj+1];
            }
          }
          i++;
          j--;
          kj--;
        }      
        
        first--;
        last++;
      }                    
    }
    // end of removing knot n=k[a]
                  
    // load the knot ua
    if (a != d)
      for (i = 0; i < ph-oldr; i++)
      {
        ik[kind] = ua;
        kind++;
      }

    // load ctrl pts into ic
    for (j = lbz; j <= rbz; j++)
    {
      for (ii = 0; ii < mc; ii++)
        ictrl[ii][cind] = ebpts[ii][j];
      cind++;
    }
    
    if (b < m)
    {
      // setup for next pass thru loop
      for (j = 0; j < r; j++)
        for (ii = 0; ii < mc; ii++)
          bpts[ii][j] = Nextbpts[ii][j];
      for (j = r; j <= d; j++)
        for (ii = 0; ii < mc; ii++)
          bpts[ii][j] = ctrl[ii][b-d+j];
      a = b;
      b++;
      ua = ub;
    }
    else
      // end knot
      for (i = 0; i <= ph; i++)
        ik[kind+i] = ub;
  }                  
  // end while loop   
  
  *nh = mh - ph - 1;

  freematrix(bezalfs);
  freematrix(bpts);
  freematrix(ebpts);
  freematrix(Nextbpts);
  free(alfs);
}

static PyObject * _nurbs_bspdegelev(PyObject *self, PyObject *args)
{
	int d, mc, nc, nk, t, nh, dim[2];
	double **ctrlmat, **icmat;
	PyObject *arg2, *arg3;
	PyArrayObject *ctrl, *k, *ic, *ik;
	if(!PyArg_ParseTuple(args, "iOOi", &d, &arg2, &arg3, &t))
		return NULL;
	ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2);
	if(ctrl == NULL)
		return NULL;
	k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1);
	if(k == NULL)
		return NULL;
	mc = ctrl->dimensions[0];
	nc = ctrl->dimensions[1];
	nk = k->dimensions[0];
	dim[0] = mc;
	dim[1] = nc*(t + 1);
	ic = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE);
	ctrlmat = vec2mat(ctrl->data, mc, nc);
	icmat = vec2mat(ic->data, mc, nc*(t + 1));
	dim[0] = (t + 1)*nk;
	ik = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
	_bspdegelev(d, ctrlmat, mc, nc, (double *)k->data, nk, t, &nh, icmat, (double *)ik->data);
	free(icmat);
	free(ctrlmat);
	Py_DECREF(ctrl);
	Py_DECREF(k);
	return Py_BuildValue("(OOi)", (PyObject *)ic, (PyObject *)ik, nh);
}

static char bspbezdecom__doc__[] =
"Decompose a B-Spline to Bezier segments.\n\
\n\
INPUT:\n\
\n\
 n,p,U,Pw\n\
\n\
OUTPUT:\n\
\n\
 Qw\n\
\n\
Modified version of Algorithm A5.6 from 'The NURBS BOOK' pg173.\n\
\n";

static void _bspbezdecom(int d, double **ctrl, int mc, int nc, double *k, int nk, 
               double **ictrl)
{
  int i, j, s, m, r, a, b, mul, n, nb, ii, save, q;
  double ua, ub, numer;
  double *alfs; 

  n = nc - 1;

  alfs = (double *) malloc(d*sizeof(double));

  m = n + d + 1;
  a = d;
  b = d+1;
  ua = k[0];
  nb = 0;
  
  // initialise first bezier seg
  for (i = 0; i <= d; i++)
    for (ii = 0; ii < mc; ii++)
      ictrl[ii][i] = ctrl[ii][i];  

  // big loop thru knot vector
  while (b < m)
  {
    i = b;
    while (b < m && k[b] == k[b+1])
      b++;

    mul = b - i + 1;
    ub = k[b];
    r = d - mul;
    
    // insert knot u(b) r times
    if (r > 0)
    {
      // insert knot to get bezier segment
      numer = ub - ua;
      for (q = d; q > mul; q--)
        alfs[q-mul-1] = numer / (k[a+q]-ua);
      for (j = 1; j <= r; j++)  
      {
        save = r - j;
        s = mul + j;            

        for (q = d; q >= s; q--)
          for (ii = 0; ii < mc; ii++)
            ictrl[ii][q+nb] = alfs[q-s]*ictrl[ii][q+nb]+(1.0-alfs[q-s])*ictrl[ii][q-1+nb];

        for (ii = 0; ii < mc; ii++)
          ictrl[ii][save+nb+d+1] = ictrl[ii][d]; 
      }  
    }
    // end of insert knot
    nb += d;
    if (b < m)
    {
      // setup for next pass thru loop
      for (j = r; j <= d; j++)
        for (ii = 0; ii < mc; ii++)
          ictrl[ii][j+nb] = ctrl[ii][b-d+j];
      a = b;
      b++;
      ua = ub;
    }
  }                 
  // end while loop   
  
  free(alfs);
}

static PyObject * _nurbs_bspbezdecom(PyObject *self, PyObject *args)
{
	int i, b, c, d, mc, nc, nk, m,  dim[2];
	double **ctrlmat, **icmat, *ks;
	PyObject *arg2, *arg3;
	PyArrayObject *ctrl, *k, *ic;
	if(!PyArg_ParseTuple(args, "iOO", &d, &arg2, &arg3))
		return NULL;
	ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2);
	if(ctrl == NULL)
		return NULL;
	k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1);
	if(k == NULL)
		return NULL;
	mc = ctrl->dimensions[0];
	nc = ctrl->dimensions[1];
	nk = k->dimensions[0];
	
	i = d + 1;
	c = 0;
	m = nk - d - 1;
	while (i < m)
	{
		b = 1;
		while (i < m && *(double *)(k->data + i * k->strides[0]) == *(double *)(k->data + (i + 1) * k->strides[0]))
		{
			b++;
			i++;
		}
		if(b < d) 
			c = c + (d - b); 
		i++;
	}
	dim[0] = mc;
	dim[1] = nc+c;
	ic = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE);
	ctrlmat = vec2mat(ctrl->data, mc, nc);
	icmat = vec2mat(ic->data, mc, nc+c);
	_bspbezdecom(d, ctrlmat, mc, nc, (double *)k->data, nk, icmat);
	free(icmat);
	free(ctrlmat); 
	Py_DECREF(ctrl);
	Py_DECREF(k); 
	return Py_BuildValue("O", ic);
}

static PyMethodDef _nurbs_methods[] =
{
	{"bincoeff", _nurbs_bincoeff, METH_VARARGS, bincoeff__doc__},
	{"bspeval", _nurbs_bspeval, METH_VARARGS, bspeval__doc__},
	{"bspdeval", _nurbs_bspdeval, METH_VARARGS, bspdeval__doc__},
	{"bspkntins", _nurbs_bspkntins, METH_VARARGS, bspkntins__doc__},
	{"bspdegelev", _nurbs_bspdegelev, METH_VARARGS, bspdegelev__doc__},
	{"bspbezdecom", _nurbs_bspbezdecom, METH_VARARGS, bspbezdecom__doc__},
	{NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_nurbs_()
{
	PyObject *m;
	m = Py_InitModule3("_nurbs_", _nurbs_methods, _nurbs_module__doc__);
	import_array(); /* Get access to numpy array API */
}

/* End */

	
