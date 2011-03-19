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

//
// Part of this module was inspired by the Nurbs toolbox Python port by 
// Runar Tenfjord (http://www.aria.uklinux.net/nurbs.php3)
//
// It was optimized and adapted to numpy by Benedict Verhegghe
//

#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>

/****** INTERNAL FUNCTIONS (not callable from Python ********/

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

/* Dot product of two vectors of length n */
/* ia and ib are the strides of the elements addressed starting from a, b */ 
static double dotprod(double *a, int ia, double *b, int ib, int n)
{
  int i;
  double t;
  t = 0.0;
  for (i=0; i<n; i++) {
    t += (*a)*(*b);
    a += ia;
    b += ib;
  }
  return t;
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

static void print_mat(double *mat,int nrows,int ncols)
{
  int i,j;
  for (i=0;  i<nrows; i++) {
    for (j=0; j<ncols; j++) printf(" %e",mat[i*ncols+j]);
    printf("\n");
  } 
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

static double _binomial(int n, int k)
{
  return floor(0.5+exp(_factln(n)-_factln(k)-_factln(n-k)));
}


/* _horner

Compute the value of a polynomial using Horner's rule.

Input:
- a: double(n+1), coefficients of the polynomial, starting
     from lowest degree
- n: int, degree of the polynomial
- u: double, parametric value where the polynomial is evaluated

Returns:
double, the value of the polynomial

Algorithm A1.1 from 'The NURBS Book' p7.
*/

static double _horner(double *a, int n, double u)
{
  double c = a[n];
  int i;
  for (i = n-1; i>=0; --i) c = c * u + a[i];
  return c;
}


/* /\* _horner */

/* Compute the value of a polynom using Horner's rule. */

/* Input: */
/* - a: double(n+1,nd), nd-dimensional coefficients of the polynom, starting  */
/*      from lowest degree */
/* - n: int, degree of the polynom */
/* - nd: int, number of dimensions  */
/* - u: double(nu), parametric values where the polynom is evaluated */

/* Output: */
/* - c: double(nu,nd), nd-dimensional values of the polynom */
/* *\/ */

/* static void _horner(double *a, int n, int nd, double *u, int nu) */
/* { */
/*   int i,j,k; */
/*   double c; */
/*   for (i=0; i<nu; ++i) { */
/*     for (j=0; j<nd; ++j) { */
/*       c = a[n,j]; */
/*       for (k=n-1; i>=0; --k) c = c * u[i] + a[k,j]; */
/*       u[i,j] = c; */
/*     } */
/*   } */
/* } */

/* static char horner_doc[] = */
/* "Evaluate a polynom using Horner's rule.\n\ */
/* \n\ */
/* Params:\n\ */
/* - a: double(n+1,nd), nd-dimensional coefficients of the polynom of degree n,\n\ */
/*      starting from lowest degree\n\ */
/* - u: double(nu), parametric values where the polynom is evaluated\n\ */
/* \n\ */
/* Returns:\n\ */
/* - p: double(nu,3), nu nd-dimensonal points\n\ */
/* \n\ */
/* Extended algorithm A1.1 from 'The NURBS Book' p7.\n\ */
/* \n"; */

/* static PyObject * nurbs_horner(PyObject *self, PyObject *args) */
/* { */
/*   int n, nd, nu; */
/*   npy_intp *a_dim, *u_dim, dim[2]; */
/*   double *a, *u, *pnt; */
/*   PyObject *arg1, *arg2; */
/*   PyObject *arr1=NULL, *arr2=NULL, *ret=NULL; */

/*   if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) */
/*     return NULL; */
/*   arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY); */
/*   if(arr1 == NULL) */
/*     return NULL; */
/*   arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY); */
/*   if(arr2 == NULL) */
/*     goto fail; */

/*   /\* We suppose the dimensions are correct*\/ */
/*   a_dim = PyArray_DIMS(arr1); */
/*   u_dim = PyArray_DIMS(arr2); */
/*   n = a_dim[0]; */
/*   nd = a_dim[1]; */
/*   nu = u_dim[0]; */
/*   a = (double *)PyArray_DATA(arr1); */
/*   u = (double *)PyArray_DATA(arr2); */

/*   /\* Create the return array *\/ */
/*   dim[0] = nu; */
/*   dim[1] = nd; */
/*   ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE); */
/*   pnt = (double *)PyArray_DATA(ret); */

/*   /\* Compute *\/ */
/*   int i,j; */
/*   for (i=0; i<nu; i */
/*   _bspeval(d, ctrl, nc, mc, k, u, nu, pnt); */
/*   //printf("pnt(%d,%d)\n",nu,mc); */
/*   //print_mat(pnt,nu,mc); */

/*   /\* Clean up and return *\/ */
/*   Py_DECREF(arr1); */
/*   Py_DECREF(arr2); */
/*   Py_DECREF(arr3); */
/*   return ret; */

/*  fail: */
/*   printf("error cleanup and return\n"); */
/*   Py_XDECREF(arr1); */
/*   Py_XDECREF(arr2); */
/*   Py_XDECREF(arr3); */
/*   return NULL; */
/* } */


/* bernstein */
/*
Compute the value of a Bernstein polynomial.

Input:
- i: int, index of the polynomial
- n: int, degree of the polynomial
- u: double, parametric value where the polynomial is evaluated

Returns:
The value of the Bernstein polynomial B(i,n) at parameter value u.

Algorithm A1.2 from 'The NURBS Book' p20. 
*/

static double bernstein(int i, int n, double u)
{
  int j, k;
  double u1;
  double *temp  = (double*) malloc((n+1)*sizeof(double));
  for (j=0; i<+n; j++) temp[j] = 0.0;
  temp[n-i] = 1.0;
  u1 = 1.0-u;
  for (k=1; k<=n; k++)
    for (j=n; j<=k; j--)
      temp[j] = u1*temp[j] + u*temp[j-1];
  return temp[n];
}


/* allBernstein */
/*
Compute the value of all n-th degree Bernstein polynomials.

Input:
- n: int, degree of the polynomials
- u: double, parametric value where the polynomials are evaluated

Output:
- B: double(n+1), the value of all n-th degree Bernstein polynomials B(i,n)
at parameter value u.

Algorithm A1.3 from 'The NURBS Book' p20. 
*/

static void allBernstein(int n, double u, double *B)
{
  int j, k;
  double u1, temp, saved;
  B[0] = 1.0;
  u1 = 1.0-u;
  for (j=1; j<=n; j++) {
    saved = 0.0;
    for (k=0; k<j; k++) {
      temp = B[k];
      B[k] = saved + u1*temp;
      saved = u * temp;
    }
    B[j] = saved;
  }
}


/* findSpan */
/*
Find the knot span index of the parametric point u. 

Input:

- U: knot sequence: U[0] .. U[m]
- u: parametric value: U[0] <= u <= U[m]
- p: degree of the B-spline basis functions
- n: number of control points - 1 = m - p - 1

Returns:

- index of the knot span

Algorithm A2.1 from 'The NURBS Book' pg68.
*/
static int findSpan(double *U, double u, int p, int n)
{
  int low, high, mid;
  int cnt=0;

  // special case
  if (u == U[n+1]) return(n);
    
  // do binary search
  low = p;
  high = n + 1;
  mid = (low + high) / 2;
  while (u < U[mid] || u >= U[mid+1]) {
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

/* basisFuns */
/*
Compute the nonvanishing B-spline basis functions for index span i. 

Input:

- U: knot sequence: U[0] .. U[m]
- u: parametric value: U[0] <= u <= U[m]
- p: degree of the B-spline basis functions
- i: index of the knot span for value u (from findSpan())

Output:
- N: (p+1) values of nonzero basis functions at u

Algorithm A2.2 from 'The NURBS Book' pg70.
*/
static void basisFuns(double *U, double u, int p, int i, double *N)
{
  int j,r;
  double saved, temp;

  // work space
  double *left  = (double*) malloc((p+1)*sizeof(double));
  double *right = (double*) malloc((p+1)*sizeof(double));
  
  N[0] = 1.0;
  for (j = 1; j <= p; j++) {
    left[j]  = u - U[i+1-j];
    right[j] = U[i+j] - u;
    saved = 0.0;
    for (r = 0; r < j; r++) {
      temp = N[r] / (right[r+1] + left[j-r]);
      N[r] = saved + right[r+1] * temp;
      saved = left[j-r] * temp;
    } 
    N[j] = saved;
  }

  free(left);
  free(right);
}

/* basisDerivs */
/*
Compute the nonvanishing B-spline basis functions and their derivatives. 

Input:

- U: knot sequence: U[0] .. U[m]
- u: parametric value: U[0] <= u <= U[m]
- p: degree of the B-spline basis functions
- i: index of the knot span for value u (from findSpan())
- n: number of derivatives to compute (n <= p)

Output:
- dN: (n+1,p+1) values of the nonzero basis functions and their first n 
      derivatives at u

Algorithm A2.3 from 'The NURBS Book' pg72.
*/
static void basisDerivs(double *U, double u, int p, int i, int n, double *dN)
{
  int j,k,r,s1,s2,rk,pk,j1,j2;
  double temp, saved, der;
  double **ndu, *a, *left, *right;

  ndu = matrix(p+1, p+1);
  a = (double *) malloc(2*(p+1)*sizeof(double));
  left = (double *) malloc((p+1)*sizeof(double));
  right = (double *) malloc((p+1)*sizeof(double));

  ndu[0][0] = 1.0;
  for (j=1; j<=p; j++) {
    left[j] = u - U[i+1-j];
    right[j] = U[i+j]-u;
    saved = 0.0;
    for (r=0; r<j; r++) {
      /* Lower triangle */
      ndu[j][r] = right[r+1] + left[j-r];
      temp = ndu[r][j-1]/ndu[j][r];
      /* Upper Triangle */
      ndu[r][j] = saved + right[r+1]*temp;
      saved = left[j-r]*temp;
    }
    ndu[j][j] = saved;
  }
  /* Load the basis functions */
  for (j=0; j<=p; j++) dN[j] = ndu[j][p];

  /* Compute the derivatives (Eq. 2.9) */
  for (r=0; r<=p; r++) {   /* Loop over function index */
    s1 = 0; s2 = p+1;      /* Alternate rows in array a */
    a[0] = 1.0;

    /* Loop to compute kth derivative */
    for (k=1; k<=n; k++) {
      der = 0.0;
      rk = r-k;  pk = p-k;
      if (r >= k) {
        a[s2] = a[s1] / ndu[pk+1][rk];
        der = a[s2] * ndu[rk][pk];
      }
      if (rk >= -1) j1 = 1;
      else j1 = -rk;
      if (r-1 <= pk) j2 = k-1;
      else j2 = p-r;
      for (j=j1; j<=j2; j++) {
        a[s2+j] = (a[s1+j] - a[s1+j-1]) / ndu[pk+1][rk+j];
        der += a[s2+j] * ndu[rk+j][pk];
      }
      if (r <= pk) {
        a[s2+k] = -a[s1+k-1] / ndu[pk+1][r];
        der += a[s2+k] * ndu[r][pk];
      }
      dN[k*(p+1)+r] = der;
      /* Switch rows */
      j = s1; s1 = s2; s2 = j;
    }
  }
  
  /* Multiply by the correct factors */
  r = p;
  for (k=1; k<=n; k++) {
    for (j=0; j<=p; j++) dN[k*(p+1)+j] *= r;
    r *= (p-k);
  }

  freematrix(ndu);
  free(a);
  free(left);
  free(right);
}


/* curvePoints */
/*
Compute a point on a B-spline curve. 

Input:

- p: degree of the B-spline
- P: control points P(nc,nd)
- nc: number of control points
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]
- u: parametric values: U[0] <= ui <= U[m]
- nu: number of parametric values

Output:
- pnt: (nu,nd) points on the B-spline

Modified algorithm A3.1 from 'The NURBS Book' pg82.
*/
static void curvePoints(int p, double *P, int nc, int nd, double *U, double *u, int nu, double *pnt)
{
  int i, j, s, t;
  
  /* space for the basis functions */
  double *N = (double*) malloc((p+1)*sizeof(double));

  /* for each parametric point j */
  for (j = 0; j < nu; j++) {

    /* find the span index of u[j] */
    s = findSpan(U,u[j],p,nc-1);
    basisFuns(U,u[j],p,s,N);

    t = (s-p) * nd;
    for (i = 0; i < nd; i++) {
      pnt[j*nd+i] = dotprod(N,1,P+t+i,nd,p+1);
    }
  }
  free(N);
} 


/* curveDerivs */
/*
Compute derivatives of a B-spline curve. 

Input:

- p: degree of the B-spline
- P: control points P(nc,nd)
- nc: number of control points
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]
- u: parametric values: U[0] <= ui <= U[m]
- nu: number of parametric values
- n: number of derivatives to compute

Output:
- pnt: (n+1,nu,nd) points and derivatives on the B-spline

Modified algorithm A3.2 from 'The NURBS Book' pg93.
*/
static void curveDerivs(int p, int n, double *P, int nc, int nd, double *U, double *u, int nu, double *pnt)
{
  int i, j, l, s, t;

  /* number of nonzero derivatives to compute */
  int du = min(p,n);

  /* space for the basis functions and derivs (du+1,p+1) */
  double *dN = (double *) malloc((du+1)*(p+1)*sizeof(double));
  for (i = 0; i < (du+1)*(p+1); i++) dN[i] = 0.0;

  /* for each parametric point r */
  for (j = 0; j < nu; j++) {
    s = findSpan(U,u[j],p,nc-1);
    basisDerivs(U,u[j],p,s,du,dN);

    /* for each nonzero dervative */
    for (l = 0; l <= du; l++) {

      t = (s-p) * nd;
      for (i = 0; i < nd; i++) {
	pnt[(l*nu+j)*nd+i] = dotprod(dN+l*(p+1),1,P+t+i,nd,p+1);
      }
    }
  }
  /* clear remainder */
  for (l = du+1; l <= n; l++)
    for (j = 0; j < nu; j++)
      for (i = 0; i < nd; i++)
	pnt[(l*nu+j)*nd+i] = 0.0;
  free(dN);
}

static void _bspkntins(int p, double **ctrl, int nd, int nc, double *k, int nk, 
              double *u, int nu, double **ictrl, double *ik)
{
  int a, b, r, l, i, j, m, n, s, q, ind;
  double alfa;

  n = nc - 1;
  r = nu - 1;

  m = n + p + 1;
  a = findSpan(k,u[0],p,n);
  b = findSpan(k,u[r],p,n);
  ++b;

  for (q = 0; q < nd; q++)
  {
    for (j = 0; j <= a-p; j++) ictrl[q][j] = ctrl[q][j];
    for (j = b-1; j <= n; j++) ictrl[q][j+r+1] = ctrl[q][j];
  }
  for (j = 0; j <= a; j++)   ik[j] = k[j];
  for (j = b+p; j <= m; j++) ik[j+r+1] = k[j];

  i = b + p - 1;
  s = b + p + r;
  for (j = r; j >= 0; j--)
  {
    while (u[j] <= k[i] && i > a)
    {
      for (q = 0; q < nd; q++)
        ictrl[q][s-p-1] = ctrl[q][i-p-1];
      ik[s] = k[i];
      --s;
      --i;
    }
    for (q = 0; q < nd; q++)
      ictrl[q][s-p-1] = ictrl[q][s-p];
    for (l = 1; l <= p; l++)
    {
      ind = s - p + l;
      alfa = ik[s+l] - u[j];
      if (fabs(alfa) == 0.0)
        for (q = 0; q < nd; q++)
          ictrl[q][ind-1] = ictrl[q][ind];
      else
      {
        alfa /= (ik[s+l] - k[i-p+l]);
        for (q = 0; q < nd; q++)
          ictrl[q][ind-1] = alfa*ictrl[q][ind-1]+(1.0-alfa)*ictrl[q][ind];
      }
    }

    ik[s] = u[j];
    --s;
  }
}

/* static PyObject * nurbs_bspkntins(PyObject *self, PyObject *args) */
/* { */
/*   int p, nd, nc, nk, nu, dim[2]; */
/*   double **ctrlmat, **icmat; */
/*   PyObject *arg2, *arg3, *arg4; */
/*   PyArrayObject *ctrl, *k, *u, *ic, *ik; */
/*   if(!PyArg_ParseTuple(args, "iOOO", &p, &arg2, &arg3, &arg4)) */
/*     return NULL; */
/*   ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2); */
/*   if(ctrl == NULL) */
/*     return NULL; */
/*   k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1); */
/*   if(k == NULL) */
/*     return NULL; */
/*   u = (PyArrayObject *) PyArray_ContiguousFromObject(arg4, PyArray_DOUBLE, 1, 1); */
/*   if(u == NULL) */
/*     return NULL; */
/*   nd = ctrl->dimensions[0]; */
/*   nc = ctrl->dimensions[1]; */
/*   nk = k->dimensions[0]; */
/*   nu = u->dimensions[0]; */
/*   dim[0] = nd; */
/*   dim[1] = nc + nu; */
/*   ic = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE); */
/*   ctrlmat = vec2mat(ctrl->data, nd, nc); */
/*   icmat = vec2mat(ic->data, nd, nc + nu); */
/*   dim[0] = nk + nu; */
/*   ik = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE); */
/*   _bspkntins(p, ctrlmat, nd, nc, (double *)k->data, nk, (double *)u->data, nu, icmat, (double *)ik->data); */
/*   free(icmat); */
/*   free(ctrlmat); */
/*   Py_DECREF(ctrl); */
/*   Py_DECREF(k); */
/*   Py_DECREF(u); */
/*   return Py_BuildValue("(OO)", (PyObject *)ic, (PyObject *)ik); */
/* } */

/* static char bspdegelev_doc[] = */
/* "Degree elevate a B-Spline t times.\n\ */
/* \n\ */
/* INPUT:\n\ */
/* \n\ */
/*  n,p,U,Pw,t\n\ */
/* \n\ */
/* OUTPUT:\n\ */
/* \n\ */
/*  nh,Uh,Qw\n\ */
/* \n\ */
/* Modified version of Algorithm A5.9 from 'The NURBS BOOK' pg206.\n\ */
/* \n"; */

/* static void _bspdegelev(int p, double **ctrl, int nd, int nc, double *k, int nk,  */
/*                int t, int *nh, double **ictrl, double *ik) */
/* { */
/*   int i, j, q, s, m, ph, ph2, mpi, mh, r, a, b, cind, oldr, mul; */
/*   int n, lbz, rbz, save, tr, kj, first, kind, last, bet, ii; */
/*   double inv, ua, ub, numer, den, alf, gam; */
/*   double **bezalfs, **bpts, **ebpts, **Nextbpts, *alfs;  */

/*   n = nc - 1; */

/*   bezalfs = matrix(d+1,p+t+1); */
/*   bpts = matrix(nd,p+1); */
/*   ebpts = matrix(nd,p+t+1); */
/*   Nextbpts = matrix(nd,p); */
/*   alfs = (double *) malloc(p*sizeof(double)); */

/*   m = n + p + 1; */
/*   ph = p + t; */
/*   ph2 = ph / 2; */

/*   // compute bezier degree elevation coefficeients   */
/*   bezalfs[0][0] = bezalfs[p][ph] = 1.0; */

/*   for (i = 1; i <= ph2; i++) */
/*   { */
/*     inv = 1.0 / _binomial(ph,i); */
/*     mpi = min(p,i); */
    
/*     for (j = max(0,i-t); j <= mpi; j++) */
/*       bezalfs[j][i] = inv * _binomial(p,j) * _binomial(t,i-j); */
/*   }     */
  
/*   for (i = ph2+1; i <= ph-1; i++) */
/*   { */
/*     mpi = min(p, i); */
/*     for (j = max(0,i-t); j <= mpi; j++) */
/*       bezalfs[j][i] = bezalfs[p-j][ph-i]; */
/*   }        */

/*   mh = ph; */
/*   kind = ph+1; */
/*   r = -1; */
/*   a = p; */
/*   b = p+1; */
/*   cind = 1; */
/*   ua = k[0]; */
/*   for (ii = 0; ii < nd; ii++) */
/*     ictrl[ii][0] = ctrl[ii][0]; */
  
/*   for (i = 0; i <= ph; i++) */
/*     ik[i] = ua; */
    
/*   // initialise first bezier seg */
/*   for (i = 0; i <= p; i++) */
/*     for (ii = 0; ii < nd; ii++) */
/*       bpts[ii][i] = ctrl[ii][i];   */

/*   // big loop thru knot vector */
/*   while (b < m) */
/*   { */
/*     i = b; */
/*     while (b < m && k[b] == k[b+1]) */
/*       b++; */

/*     mul = b - i + 1; */
/*     mh += mul + t; */
/*     ub = k[b]; */
/*     oldr = r; */
/*     r = p - mul; */
    
/*     // insert knot u(b) r times */
/*     if (oldr > 0) */
/*       lbz = (oldr+2) / 2; */
/*     else */
/*       lbz = 1; */

/*     if (r > 0) */
/*       rbz = ph - (r+1)/2; */
/*     else */
/*       rbz = ph;   */

/*     if (r > 0) */
/*     { */
/*       // insert knot to get bezier segment */
/*       numer = ub - ua; */
/*       for (q = p; q > mul; q--) */
/*         alfs[q-mul-1] = numer / (k[a+q]-ua); */
/*       for (j = 1; j <= r; j++)   */
/*       { */
/*         save = r - j; */
/*         s = mul + j;             */

/*         for (q = p; q >= s; q--) */
/*           for (ii = 0; ii < nd; ii++) */
/*             bpts[ii][q] = alfs[q-s]*bpts[ii][q]+(1.0-alfs[q-s])*bpts[ii][q-1]; */

/*         for (ii = 0; ii < nd; ii++) */
/*           Nextbpts[ii][save] = bpts[ii][p]; */
/*       }   */
/*     } */
/*     // end of insert knot */

/*     // degree elevate bezier */
/*     for (i = lbz; i <= ph; i++) */
/*     { */
/*       for (ii = 0; ii < nd; ii++) */
/*         ebpts[ii][i] = 0.0; */
/*       mpi = min(p, i); */
/*       for (j = max(0,i-t); j <= mpi; j++) */
/*         for (ii = 0; ii < nd; ii++) */
/*           ebpts[ii][i] = ebpts[ii][i] + bezalfs[j][i]*bpts[ii][j]; */
/*     } */
/*     // end of degree elevating bezier */

/*     if (oldr > 1) */
/*     { */
/*       // must remove knot u=k[a] oldr times */
/*       first = kind - 2; */
/*       last = kind; */
/*       den = ub - ua; */
/*       bet = (ub-ik[kind-1]) / den; */
      
/*       // knot removal loop */
/*       for (tr = 1; tr < oldr; tr++) */
/*       {         */
/*         i = first; */
/*         j = last; */
/*         kj = j - kind + 1; */
/*         while (j - i > tr) */
/*         { */
/*           // loop and compute the new control points */
/*           // for one removal step */
/*           if (i < cind) */
/*           { */
/*             alf = (ub-ik[i])/(ua-ik[i]); */
/*             for (ii = 0; ii < nd; ii++) */
/*               ictrl[ii][i] = alf * ictrl[ii][i] + (1.0-alf) * ictrl[ii][i-1]; */
/*           }         */
/*           if (j >= lbz) */
/*           { */
/*             if (j-tr <= kind-ph+oldr) */
/*             {   */
/*               gam = (ub-ik[j-tr]) / den; */
/*               for (ii = 0; ii < nd; ii++) */
/*                 ebpts[ii][kj] = gam*ebpts[ii][kj] + (1.0-gam)*ebpts[ii][kj+1]; */
/*             } */
/*             else */
/*             { */
/*               for (ii = 0; ii < nd; ii++) */
/*                 ebpts[ii][kj] = bet*ebpts[ii][kj] + (1.0-bet)*ebpts[ii][kj+1]; */
/*             } */
/*           } */
/*           i++; */
/*           j--; */
/*           kj--; */
/*         }       */
        
/*         first--; */
/*         last++; */
/*       }                     */
/*     } */
/*     // end of removing knot n=k[a] */
                  
/*     // load the knot ua */
/*     if (a != p) */
/*       for (i = 0; i < ph-oldr; i++) */
/*       { */
/*         ik[kind] = ua; */
/*         kind++; */
/*       } */

/*     // load ctrl pts into ic */
/*     for (j = lbz; j <= rbz; j++) */
/*     { */
/*       for (ii = 0; ii < nd; ii++) */
/*         ictrl[ii][cind] = ebpts[ii][j]; */
/*       cind++; */
/*     } */
    
/*     if (b < m) */
/*     { */
/*       // setup for next pass thru loop */
/*       for (j = 0; j < r; j++) */
/*         for (ii = 0; ii < nd; ii++) */
/*           bpts[ii][j] = Nextbpts[ii][j]; */
/*       for (j = r; j <= p; j++) */
/*         for (ii = 0; ii < nd; ii++) */
/*           bpts[ii][j] = ctrl[ii][b-p+j]; */
/*       a = b; */
/*       b++; */
/*       ua = ub; */
/*     } */
/*     else */
/*       // end knot */
/*       for (i = 0; i <= ph; i++) */
/*         ik[kind+i] = ub; */
/*   }                   */
/*   // end while loop    */
  
/*   *nh = mh - ph - 1; */

/*   freematrix(bezalfs); */
/*   freematrix(bpts); */
/*   freematrix(ebpts); */
/*   freematrix(Nextbpts); */
/*   free(alfs); */
/* } */

/* static PyObject * nurbs_bspdegelev(PyObject *self, PyObject *args) */
/* { */
/* 	int p, nd, nc, nk, t, nh, dim[2]; */
/* 	double **ctrlmat, **icmat; */
/* 	PyObject *arg2, *arg3; */
/* 	PyArrayObject *ctrl, *k, *ic, *ik; */
/* 	if(!PyArg_ParseTuple(args, "iOOi", &d, &arg2, &arg3, &t)) */
/* 		return NULL; */
/* 	ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2); */
/* 	if(ctrl == NULL) */
/* 		return NULL; */
/* 	k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1); */
/* 	if(k == NULL) */
/* 		return NULL; */
/* 	nd = ctrl->dimensions[0]; */
/* 	nc = ctrl->dimensions[1]; */
/* 	nk = k->dimensions[0]; */
/* 	dim[0] = nd; */
/* 	dim[1] = nc*(t + 1); */
/* 	ic = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE); */
/* 	ctrlmat = vec2mat(ctrl->data, nd, nc); */
/* 	icmat = vec2mat(ic->data, nd, nc*(t + 1)); */
/* 	dim[0] = (t + 1)*nk; */
/* 	ik = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE); */
/* 	_bspdegelev(p, ctrlmat, nd, nc, (double *)k->data, nk, t, &nh, icmat, (double *)ik->data); */
/* 	free(icmat); */
/* 	free(ctrlmat); */
/* 	Py_DECREF(ctrl); */
/* 	Py_DECREF(k); */
/* 	return Py_BuildValue("(OOi)", (PyObject *)ic, (PyObject *)ik, nh); */
/* } */

/* static char bspbezdecom_doc[] = */
/* "Decompose a B-Spline to Bezier segments.\n\ */
/* \n\ */
/* INPUT:\n\ */
/* \n\ */
/*  n,p,U,Pw\n\ */
/* \n\ */
/* OUTPUT:\n\ */
/* \n\ */
/*  Qw\n\ */
/* \n\ */
/* Modified version of Algorithm A5.6 from 'The NURBS BOOK' pg173.\n\ */
/* \n"; */

/* static void _bspbezdecom(int p, double **ctrl, int nd, int nc, double *k, int nk,  */
/*                double **ictrl) */
/* { */
/*   int i, j, s, m, r, a, b, mul, n, nb, ii, save, q; */
/*   double ua, ub, numer; */
/*   double *alfs;  */

/*   n = nc - 1; */

/*   alfs = (double *) malloc(d*sizeof(double)); */

/*   m = n + p + 1; */
/*   a = p; */
/*   b = p+1; */
/*   ua = k[0]; */
/*   nb = 0; */
  
/*   // initialise first bezier seg */
/*   for (i = 0; i <= p; i++) */
/*     for (ii = 0; ii < nd; ii++) */
/*       ictrl[ii][i] = ctrl[ii][i];   */

/*   // big loop thru knot vector */
/*   while (b < m) */
/*   { */
/*     i = b; */
/*     while (b < m && k[b] == k[b+1]) */
/*       b++; */

/*     mul = b - i + 1; */
/*     ub = k[b]; */
/*     r = p - mul; */
    
/*     // insert knot u(b) r times */
/*     if (r > 0) */
/*     { */
/*       // insert knot to get bezier segment */
/*       numer = ub - ua; */
/*       for (q = p; q > mul; q--) */
/*         alfs[q-mul-1] = numer / (k[a+q]-ua); */
/*       for (j = 1; j <= r; j++)   */
/*       { */
/*         save = r - j; */
/*         s = mul + j;             */

/*         for (q = p; q >= s; q--) */
/*           for (ii = 0; ii < mc; ii++) */
/*             ictrl[ii][q+nb] = alfs[q-s]*ictrl[ii][q+nb]+(1.0-alfs[q-s])*ictrl[ii][q-1+nb]; */

/*         for (ii = 0; ii < mc; ii++) */
/*           ictrl[ii][save+nb+p+1] = ictrl[ii][p];  */
/*       }   */
/*     } */
/*     // end of insert knot */
/*     nb += p; */
/*     if (b < m) */
/*     { */
/*       // setup for next pass thru loop */
/*       for (j = r; j <= p; j++) */
/*         for (ii = 0; ii < mc; ii++) */
/*           ictrl[ii][j+nb] = ctrl[ii][b-p+j]; */
/*       a = b; */
/*       b++; */
/*       ua = ub; */
/*     } */
/*   }                  */
/*   // end while loop    */
  
/*   free(alfs); */
/* } */

/* static PyObject * nurbs_bspbezdecom(PyObject *self, PyObject *args) */
/* { */
/* 	int i, b, c, p, mc, nc, nk, m,  dim[2]; */
/* 	double **ctrlmat, **icmat, *ks; */
/* 	PyObject *arg2, *arg3; */
/* 	PyArrayObject *ctrl, *k, *ic; */
/* 	if(!PyArg_ParseTuple(args, "iOO", &p, &arg2, &arg3)) */
/* 		return NULL; */
/* 	ctrl = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 2, 2); */
/* 	if(ctrl == NULL) */
/* 		return NULL; */
/* 	k = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE, 1, 1); */
/* 	if(k == NULL) */
/* 		return NULL; */
/* 	mc = ctrl->dimensions[0]; */
/* 	nc = ctrl->dimensions[1]; */
/* 	nk = k->dimensions[0]; */
	
/* 	i = p + 1; */
/* 	c = 0; */
/* 	m = nk - p - 1; */
/* 	while (i < m) */
/* 	{ */
/* 		b = 1; */
/* 		while (i < m && *(double *)(k->data + i * k->strides[0]) == *(double *)(k->data + (i + 1) * k->strides[0])) */
/* 		{ */
/* 			b++; */
/* 			i++; */
/* 		} */
/* 		if(b < p)  */
/* 			c = c + (p - b);  */
/* 		i++; */
/* 	} */
/* 	dim[0] = mc; */
/* 	dim[1] = nc+c; */
/* 	ic = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE); */
/* 	ctrlmat = vec2mat(ctrl->data, mc, nc); */
/* 	icmat = vec2mat(ic->data, mc, nc+c); */
/* 	_bspbezdecom(d, ctrlmat, mc, nc, (double *)k->data, nk, icmat); */
/* 	free(icmat); */
/* 	free(ctrlmat);  */
/* 	Py_DECREF(ctrl); */
/* 	Py_DECREF(k);  */
/* 	return Py_BuildValue("O", ic); */
/* } */


/****** EXPORTED FUNCTIONS (callable from Python ********/

static char _doc_[] = "nurbs_ module. Version 0.1\n\
\n\
This module implements low level NURBS functions for pyFormex.\n\
\n";


static char binomial_doc[] =
"Computes the binomial coefficient.\n\
\n\
 ( n )      n!\n\
 (   ) = --------\n\
 ( k )   k!(n-k)!\n\
\n\
 Algorithm from 'Numerical Recipes in C, 2nd Edition' pg215.\n";

static PyObject * nurbs_binomial(PyObject *self, PyObject *args)
{
  int n, k;
  double ret;
  if(!PyArg_ParseTuple(args, "ii", &n, &k))
    return NULL;
  ret = _binomial(n, k);
  return Py_BuildValue("d",ret);
}


static char allBernstein_doc[] =
"Compute the value of all n-th degree Bernstein polynomials.\n\
\n\
Input:\n\
- n: int, degree of the polynomials\n\
- u: double, parametric value where the polynomials are evaluated\n\
\n\
Returns:\n\
double(n+1), the value of all n-th degree Bernstein polynomials B(i,n)\n\
at parameter value u.\n\
\n\
Algorithm A1.3 from The NURBS Book.\n";

static PyObject * nurbs_allBernstein(PyObject *self, PyObject *args)
{
  int n;
  npy_intp dim[1];
  double u, *B;
  PyObject *ret=NULL;

  if (!PyArg_ParseTuple(args, "id", &n, &u))
    return NULL;

  /* Create the return array */
  dim[0] = n+1;
  ret = PyArray_SimpleNew(1,dim, NPY_DOUBLE);
  B = (double *)PyArray_DATA(ret);

  /* Compute */
  allBernstein(n,u,B);

  /* Return */
  return ret;
}


static char curvePoints_doc[] =
"Compute a point on a B-spline curve.\n\
\n\
Input:\n\
\n\
- p: degree of the B-spline\n\
- P: control points P(nc,nd)\n\
- nc: number of control points\n\
- nd: dimension of the points (3 or 4)\n\
- U: knot sequence: U[0] .. U[m]\n\
- u: parametric values: U[0] <= ui <= U[m]\n\
- nu: number of parametric values\n\
\n\
Output:\n\
- pnt: (nu,nd) points on the B-spline\n\
\n\
Modified algorithm A3.1 from 'The NURBS Book' pg82.\n\
\n";

static PyObject * nurbs_curvePoints(PyObject *self, PyObject *args)
{
  int p, nd, nc, nk, nu;
  npy_intp *ctrl_dim, *k_dim, *u_dim, dim[2];
  double *ctrl, *k, *u, *pnt;
  PyObject *a1, *a2, *a3;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret=NULL;

  if (!PyArg_ParseTuple(args, "iOOO", &p, &a1, &a2, &a3))
    return NULL;
  arr1 = PyArray_FROM_OTF(a1, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr1 == NULL)
    return NULL;
  arr2 = PyArray_FROM_OTF(a2, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr2 == NULL)
    goto fail;
  arr3 = PyArray_FROM_OTF(a3, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr3 == NULL)
    goto fail;

  ctrl_dim = PyArray_DIMS(arr1);
  k_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  nc = ctrl_dim[0];
  nd = ctrl_dim[1];
  nk = k_dim[0];
  nu = u_dim[0];
  ctrl = (double *)PyArray_DATA(arr1);
  k = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);

  /* Create the return array */
  dim[0] = nu;
  dim[1] = nd;
  ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  pnt = (double *)PyArray_DATA(ret);

  /* Compute */
  curvePoints(p, ctrl, nc, nd, k, u, nu, pnt);

  /* Clean up and return */
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


static char curveDerivs_doc[] =
"Compute derivatives of a B-spline curve.\n\
\n\
Input:\n\
\n\
- p: degree of the B-spline\n\
- P: control points P(nc,nd)\n\
- nc: number of control points\n\
- nd: dimension of the points (3 or 4)\n\
- U: knot sequence: U[0] .. U[m]\n\
- u: parametric values: U[0] <= ui <= U[m]\n\
- nu: number of parametric values\n\
- n: number of derivatives to compute\n\
\n\
Output:\n\
- pnt: (n+1,nu,nd) points and derivatives on the B-spline\n\
\n\
Modified algorithm A3.2 from 'The NURBS Book' pg93.\n\
\n";

static PyObject * nurbs_curveDerivs(PyObject *self, PyObject *args)
{
  int p, nd, nc, nk, nu, n;
  npy_intp *ctrl_dim, *k_dim, *u_dim, dim[3];
  double *ctrl, *k, *u, *pnt;
  PyObject *a1, *a2, *a3;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret=NULL;

  if(!PyArg_ParseTuple(args, "iiOOO", &p, &n, &a1, &a2, &a3))
    return NULL;
  arr1 = PyArray_FROM_OTF(a1, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr1 == NULL)
    return NULL;
  arr2 = PyArray_FROM_OTF(a2, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr2 == NULL)
    goto fail;
  arr3 = PyArray_FROM_OTF(a3, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr3 == NULL)
    goto fail;

  ctrl_dim = PyArray_DIMS(arr1);
  k_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  nc = ctrl_dim[0];
  nd = ctrl_dim[1];
  nk = k_dim[0];
  nu = u_dim[0];
  ctrl = (double *)PyArray_DATA(arr1);
  k = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);

  /* Create the return array */
  dim[0] = n+1;
  dim[1] = nu;
  dim[2] = nd;
  ret = PyArray_SimpleNew(3,dim, NPY_DOUBLE);
  pnt = (double *)PyArray_DATA(ret);

  /* Compute */
  curveDerivs(p, n, ctrl, nc, nd, k, u, nu, pnt);

  /* Clean up and return */
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

static char bspkntins_doc[] =
"Insert Knot into a B-Spline.\n\
\n\
INPUT:\n\
\n\
 p - spline degree       integer\n\
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

static PyMethodDef _methods_[] =
{
	{"binomial", nurbs_binomial, METH_VARARGS, binomial_doc},
	{"allBernstein", nurbs_allBernstein, METH_VARARGS, allBernstein_doc},
	{"curvePoints", nurbs_curvePoints, METH_VARARGS, curvePoints_doc},
	{"curveDerivs", nurbs_curveDerivs, METH_VARARGS, curveDerivs_doc},
	/* {"bspkntins", nurbs_bspkntins, METH_VARARGS, bspkntins_doc}, */
	/* {"bspdegelev", nurbs_bspdegelev, METH_VARARGS, bspdegelev_doc}, */
	/* {"bspbezdecom", nurbs_bspbezdecom, METH_VARARGS, bspbezdecom_doc}, */
	{NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initnurbs_(void)
{
  (void) Py_InitModule3("nurbs_", _methods_, _doc_);
  import_array(); /* Get access to numpy array API */
}

/* End */

	
