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

//
// This module is partly inspired by the Nurbs toolbox Python port by 
// Runar Tenfjord (http://www.aria.uklinux.net/nurbs.php3)
//

#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>

static char __doc__[] = "nurbs_ module\n\
\n\
This module provides accelerated versions of the pyFormex NURBS\n\
functions.\n\
\n";

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

/* Distance between two points in n dimensions */
/* p and q are n-dimensional points. */
static double distance4d(double *a, double *b, int n)
{
  int i;
  double s,t;
  t = 0.0;
  for (i=0; i<n; i++) {
    s = (*a)-(*b);
    t += s*s;
    ++a;
    ++b;
  }
  return t;
}
  

/* Turn an array into a matrix */
/* An array here is a contiguous memory space of (nrows*ncols) doubles,
   stored in row first order. This function creates an array of pointers
   to the start of each row. As a result the array elements can be addressed
   as p[i][j], and operations can be done on the whole row.
*/
static double **matrix(double*a, int nrows, int ncols) 
{
  int row;
  double **mat;

  mat = (double**) malloc (nrows*sizeof(double*));
  mat[0] = a;
  for (row = 1; row < nrows; row++)
    mat[row] = mat[row-1] + ncols;  
  return mat;
}

static double **newmatrix(int nrows, int ncols) 
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


/* _bernstein */
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

static double _bernstein(int i, int n, double u)
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


/* all_bernstein */
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

static void all_bernstein(int n, double u, double *B)
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



/* Find last occurrence of u in U */
static int find_last_occurrence(double *U, double u)
{
  int i = 0;
  while (U[i] <= u) ++i;
  return i-1;
}

/* Find multiplicity of u in U, where r is the last occurrence of u in U*/
static int find_multiplicity(double *U, double u, int r)
{
  int i = r;
  while (U[i] == u) --i;
  return r-i;
}

/* find_span */
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
static int find_span(double *U, double u, int p, int n)
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

/* basis_funs */
/*
Compute the nonvanishing B-spline basis functions for index span i. 

Input:

- U: knot sequence: U[0] .. U[m]
- u: parametric value: U[0] <= u <= U[m]
- p: degree of the B-spline basis functions
- i: index of the knot span for value u (from find_span())

Output:
- N: (p+1) values of nonzero basis functions at u

Algorithm A2.2 from 'The NURBS Book' pg70.
*/
static void basis_funs(double *U, double u, int p, int i, double *N)
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

/* basis_derivs */
/*
Compute the nonvanishing B-spline basis functions and their derivatives. 

Input:

- U: knot sequence: U[0] .. U[m]
- u: parametric value: U[0] <= u <= U[m]
- p: degree of the B-spline basis functions
- i: index of the knot span for value u (from find_span())
- n: number of derivatives to compute (n <= p)

Output:
- dN: (n+1,p+1) values of the nonzero basis functions and their first n 
      derivatives at u

Algorithm A2.3 from 'The NURBS Book' pg72.
*/
static void basis_derivs(double *U, double u, int p, int i, int n, double *dN)
{
  int j,k,r,s1,s2,rk,pk,j1,j2;
  double temp, saved, der;
  double **ndu, *a, *left, *right;

  ndu = newmatrix(p+1, p+1);
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


/********************************************************/
/************************ CURVE *************************/
/********************************************************/


/* curve_points */
/*
Compute points on a B-spline curve. 

Input:

- P: control points P(nc,nd)
- nc: number of control points
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]
- nk: number of knot values = m+1
- u: parametric values: U[0] <= ui <= U[m]
- nu: number of parametric values

Output:
- pnt: (nu,nd) points on the B-spline

Modified algorithm A3.1 from 'The NURBS Book' pg82.
*/
static void curve_points(double *P, int nc, int nd, double *U, int nk, double *u, int nu, double *pnt)
{
  int i, j, p, s, t;
  
  /* degree of the spline */
  p = nk - nc - 1;

  /* space for the basis functions */
  double *N = (double*) malloc((p+1)*sizeof(double));

  /* for each parametric point j */
  for (j=0; j<nu; ++j) {

    /* find the span index of u[j] */
    s = find_span(U,u[j],p,nc-1);
    basis_funs(U,u[j],p,s,N);

    t = (s-p) * nd;
    for (i=0; i<nd; ++i) {
      pnt[j*nd+i] = dotprod(N,1,P+t+i,nd,p+1);
    }
  }
  free(N);
} 


/* curve_derivs */
/*
Compute derivatives of a B-spline curve. 

Input:

- n: number of derivatives to compute
- P: control points P(nc,nd)
- nc: number of control points
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]
- nk: number of knot values = m+1
- u: parametric values: U[0] <= ui <= U[m]
- nu: number of parametric values

Output:
- pnt: (n+1,nu,nd) points and derivatives on the B-spline

Modified algorithm A3.2 from 'The NURBS Book' pg93.
*/
static void curve_derivs(int n, double *P, int nc, int nd, double *U, int nk, double *u, int nu, double *pnt)
{
  int i, j, l, p, s, t;

  /* degree of the spline */
  p = nk - nc - 1;

  /* number of nonzero derivatives to compute */
  int du = min(p,n);

  /* space for the basis functions and derivs (du+1,p+1) */
  double *dN = (double *) malloc((du+1)*(p+1)*sizeof(double));
  for (i = 0; i < (du+1)*(p+1); i++) dN[i] = 0.0;

  /* for each parametric point r */
  for (j = 0; j < nu; j++) {
    s = find_span(U,u[j],p,nc-1);
    basis_derivs(U,u[j],p,s,du,dN);

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



/* curve_knot_refine */
/*
Refine curve knot vector. 

Input:

- p: degree of the B-spline
- P: control points P(nc,nd)
- nc: number of control points = n+1
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p
- u: (nu) parametric values of new knots: U[0] <= u[i] <= U[m]
- nu: number of knots to insert

Output:
- newP: (nc+nu,nd) new control points
- newU: (nc+p+nu) new knot vector

Modified algorithm A5.1 from 'The NURBS Book' pg164.
*/

static void curve_knot_refine(double *P, int nc, int nd, double *U, int nk, double *u, int nu, double *newP, double *newU)
{
  int a, b, r, l, i, j, k, n, p, q, ind;
  double alfa;

  p = nk - nc - 1;
  n = nc - 1;
  r = nu - 1;

  a = find_span(U,u[0],p,n);
  b = find_span(U,u[r],p,n) + 1;

  for (j = 0; j < a-p; j++) 
    for (q=0; q<nd; q++) newP[j*nd+q] = P[j*nd+q];
  for (j = b-1; j <= n; j++)
    for (q=0; q<nd; q++) newP[(j+r+1)*nd+q] = P[j*nd+q];

  for (j = 0; j <= a; j++)   newU[j] = U[j];
  for (j = b+p; j < nk; j++) newU[j+r+1] = U[j];

  i = b + p - 1;
  k = b + p + r;
  for (j = r; j >= 0; j--) {
    while (u[j] <= U[i] && i > a) {
      for (q=0; q<nd; q++) newP[(k-p-1)*nd+q] = P[(i-p-1)*nd+q];
      newU[k] = U[i];
      --k;
      --i;
    }
    for (q=0; q<nd; q++) newP[(k-p-1)*nd+q] = newP[(k-p)*nd+q];
    for (l = 1; l <= p; l++) {
      ind = k - p + l;
      alfa = newU[k+l] - u[j];
      if (fabs(alfa) == 0.0)
        for (q=0; q<nd; q++) newP[(ind-1)*nd+q] = newP[ind*nd+q];
      else {
        alfa /= (newU[k+l] - U[i-p+l]);
        for (q=0; q<nd; q++) 
	  newP[(ind-1)*nd+q] = alfa*newP[(ind-1)*nd+q] + (1.0-alfa)*newP[ind*nd+q];
      }
    }

    newU[k] = u[j];
    --k;
  }
}

/* curve_decompose */
/*
Decompose a Nurbs curve in Bezier segments. 

Input:

- p: degree of the B-spline
- P: control points P(nc,nd)
- nc: number of control points = n+1
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p

Output:

- newP: (nb*p+1,nd) new control points
- nb: number of Bezier segments 

Modified algorithm A5.6 from 'The NURBS Book' pg173.
*/
  

static void curve_decompose(double *P, int nc, int nd, double *U, int nk, double *newP)
{
  int i, j, k, p, s, m, r, a, b, mult, n, nb, ii, save;
  double numer, alpha, *alfa;

  n = nc - 1;
  m = nk - 1;
  p = m - n - 1;

  alfa = (double *) malloc(p*sizeof(double));

  a = p;
  b = p+1;
  nb = 0;
  
  /* First bezier segment */
  for (i = 0; i < (p+1)*nd; i++) newP[i] = P[i];

  // Loop through knot vector */
  while (b < m) {
    i = b;
    while (b < m && U[b] == U[b+1]) b++;
    mult = b-i+1;
    
    if (mult < p) {
      //printf("mult at %d is %d < %d\n",b,mult,p);
      /* compute alfas */
      numer = U[b] - U[a];
      for (k = p; k > mult; k--)
        alfa[k-mult-1] = numer / (U[a+k]-U[a]);

      /* Insert knot U[b] r times */
      r = p - mult;
      for (j = 1; j <= r; j++) {
        save = r - j;
        s = mult + j; 	/* Number of new points */
        for (k = p; k >= s; k--) {
	  alpha = alfa[k-s];
	  //printf("alpha = %f\n",alpha);
          for (ii = 0; ii < nd; ii++) {
            newP[(nb+k)*nd+ii] = alpha*newP[(nb+k)*nd+ii] + (1.0-alpha)*newP[(nb+k-1)*nd+ii];
	    //printf("Setting element %d to %f\n",(nb+k)*nd+ii,newP[(nb+k)*nd+ii]);
	  }
	}
	if (b < m)
	  /* Control point of next segment */
	  for (ii = 0; ii < nd; ii++) {
	    newP[(nb+p+save)*nd+ii] = newP[(nb+p)*nd+ii];
	    //printf("Copying element %d to %f\n",(nb+p+save)*nd+ii,newP[(nb+p+save)*nd+ii]);
	  }
      }
    }
    /* Bezier segment completed */
    nb += p;
    if (b < m) {
      /* Initialize for next segment */
      for (i = r; i <= p; i++)
        for (ii = 0; ii < nd; ii++) {
          newP[(nb+i)*nd+ii] = P[(b-p+i)*nd+ii];
	  //printf("Initializing element %d to %f\n",(nb+i)*nd+ii,newP[(nb+i)*nd+ii]);
	}
      a = b;
      b++;
    }
  }
  
  free(alfa);
}



/* curve_knot_remove */
/*
Refine curve knot vector. 

Input:

- p: degree of the B-spline
- P: control points P(nc,nd)
- nc: number of control points = n+1
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p
- u: knot value to remove: U[0] <= u <= U[m]
- num: number of times to remove u
- tol: allowable tolerance for deviation of the curve. See NURBS book, p. 185

Output:
- t: actual number of times that u was removed
P and U are replaced with the new control points and knot vector

Modified algorithm A5.8 from 'The NURBS Book' pg185.
*/

static int curve_knot_remove(double *P, int nc, int nd, double *U, int nk, double u, int num, double tol)
{
  int n,m,p,ord,fout,last,first,t,off,k,i,j,ii,jj,remflag,r,s,kk;
  double alfi,alfj;

  n = nc - 1;
  m = nk - 1;
  p = m - n - 1;

  double *temp = (double*) malloc((2*p+1)*nd*sizeof(double));
  double *xtemp = (double*) malloc(nd*sizeof(double));

  r = find_last_occurrence(U,u);
  s = find_multiplicity(U,u,r);

  ord = p+1;
  fout = (2*r-s-p)/2;  /* First control point out */
  last = r-s;
  first = r-p;
  for (t=0; t<num; t++) {
    /* This loop is Eq. (5.28) */
    off = first-1; /* Diff in index between temp and P */
    for (k=0; k<nd; ++k) temp[0*nd+k] = P[off*nd+k];
    for (k=0; k<nd; ++k) temp[(last+1-off)*nd+k] = P[(last+1)*nd+k];
    i = first;
    j = last;
    ii = 1;
    jj = last-off;
    remflag = 0;
    while (j-i > t) {
      /* Compute new control points for onr removeal step */
      alfi = (u-U[i])/(U[i+ord+t]-U[i]);
      alfj = (u-U[j-t])/(U[j+ord]-U[j-t]);
      for (k=0; k<nd; ++k) temp[ii*nd+k] = (P[i*nd+k]-(1.0-alfi)*temp[(ii-1)*nd+k])/alfi;
      for (k=0; k<nd; ++k) temp[jj*nd+k] = (P[j*nd+k]-alfj*temp[(jj+1)*nd+k])/(1.0-alfj);
      ++i; ++ii;
      --j; --jj;
    }
    /* Check if knot removable */
    if (j-i < t) {
      if (distance4d(temp+(ii-1)*nd,temp+(jj+1)*nd,nd) <= tol)
	remflag = 1;
    }
    else {
      alfi = (u-U[i])/(U[i+ord+t]-U[i]);
      for (k=0; k<nd; ++k) xtemp[k] = alfi*temp[(ii+t+1)*nd+k] + (1.0-alfi)*temp[(ii-1)*nd+k];
      if (distance4d(P+i*nd,xtemp,nd) <= tol)
	remflag = 1;
    }
    if (remflag == 0)
      /* Cannot remove any more knots */
      /* Get out of for-loop */
      break;
    else {
      /* Succesful removal. Save new control points */
      i = first;
      j = last;
      while (j-i > t) {
	for (k=0; k<nd; ++k) P[i*nd+k] = temp[(i-off)*nd+k];
	for (k=0; k<nd; ++k) P[j*nd+k] = temp[(j-off)*nd+k];
	++i;
	--j;
      }
    }
    --first;
    ++last;
  }
  if (t==0) return t;
  /* Shift knots */
  for (kk=r+1; kk<= m; ++kk) U[kk-t] = U[kk];
  /* Pj thru Pi will be overwritten */
  j = fout;
  i = j;
  for (kk=1; kk<t; ++kk) {
  /*   if (kk % 2 == 1) */
  /*     ++i; */
  /*   else */
  /*     --i; */
  /* } */
  /* for (kk=i+1; kk<=n ++kk) { /\* Shift *\/ */
  /*   for (k=0; k<nd; ++k) P[j*nd+k] = P[kk*nd+k]; */
  /*   ++j; */
  }
  return t;
}

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


/* static void _bspdegelev(int p, double **P, int nd, int nc, double *U,  */
/*                int t, int *nh, double **newP, double *newU) */
/* { */
/*   int i, j, q, s, m, ph, ph2, mpi, mh, r, a, b, cind, oldr, mult; */
/*   int n, lbz, rbz, save, tr, kj, first, kind, last, bet, ii; */
/*   double inv, ua, ub, numer, den, alf, gam; */
/*   double **bezalfa, **bpts, **ebpts, **Nextbpts, *alfa;  */

/*   n = nc - 1; */

/*   bezalfa = newmatrix(d+1,p+t+1); */
/*   bpts = newmatrix(nd,p+1); */
/*   ebpts = newmatrix(nd,p+t+1); */
/*   Nextbpts = newmatrix(nd,p); */
/*   alfa = (double *) malloc(p*sizeof(double)); */

/*   m = n + p + 1; */
/*   ph = p + t; */
/*   ph2 = ph / 2; */

/*   // compute bezier degree elevation coefficeients   */
/*   bezalfa[0][0] = bezalfa[p][ph] = 1.0; */

/*   for (i = 1; i <= ph2; i++) */
/*   { */
/*     inv = 1.0 / _binomial(ph,i); */
/*     mpi = min(p,i); */
    
/*     for (j = max(0,i-t); j <= mpi; j++) */
/*       bezalfa[j][i] = inv * _binomial(p,j) * _binomial(t,i-j); */
/*   }     */
  
/*   for (i = ph2+1; i <= ph-1; i++) */
/*   { */
/*     mpi = min(p, i); */
/*     for (j = max(0,i-t); j <= mpi; j++) */
/*       bezalfa[j][i] = bezalfa[p-j][ph-i]; */
/*   }        */

/*   mh = ph; */
/*   kind = ph+1; */
/*   r = -1; */
/*   a = p; */
/*   b = p+1; */
/*   cind = 1; */
/*   ua = U[0]; */
/*   for (ii = 0; ii < nd; ii++) */
/*     newP[ii][0] = P[ii][0]; */
  
/*   for (i = 0; i <= ph; i++) */
/*     newU[i] = ua; */
    
/*   // initialise first bezier seg */
/*   for (i = 0; i <= p; i++) */
/*     for (ii = 0; ii < nd; ii++) */
/*       bpts[ii][i] = P[ii][i];   */

/*   // big loop thru knot vector */
/*   while (b < m) */
/*   { */
/*     i = b; */
/*     while (b < m && U[b] == U[b+1]) */
/*       b++; */

/*     mult = b - i + 1; */
/*     mh += mult + t; */
/*     ub = U[b]; */
/*     oldr = r; */
/*     r = p - mult; */
    
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
/*       for (q = p; q > mult; q--) */
/*         alfa[q-mult-1] = numer / (U[a+q]-ua); */
/*       for (j = 1; j <= r; j++)   */
/*       { */
/*         save = r - j; */
/*         s = mult + j;             */

/*         for (q = p; q >= s; q--) */
/*           for (ii = 0; ii < nd; ii++) */
/*             bpts[ii][q] = alfa[q-s]*bpts[ii][q]+(1.0-alfa[q-s])*bpts[ii][q-1]; */

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
/*           ebpts[ii][i] = ebpts[ii][i] + bezalfa[j][i]*bpts[ii][j]; */
/*     } */
/*     // end of degree elevating bezier */

/*     if (oldr > 1) */
/*     { */
/*       // must remove knot u=U[a] oldr times */
/*       first = kind - 2; */
/*       last = kind; */
/*       den = ub - ua; */
/*       bet = (ub-newU[kind-1]) / den; */
      
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
/*             alf = (ub-newU[i])/(ua-newU[i]); */
/*             for (ii = 0; ii < nd; ii++) */
/*               newP[ii][i] = alf * newP[ii][i] + (1.0-alf) * newP[ii][i-1]; */
/*           }         */
/*           if (j >= lbz) */
/*           { */
/*             if (j-tr <= kind-ph+oldr) */
/*             {   */
/*               gam = (ub-newU[j-tr]) / den; */
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
/*     // end of removing knot n=U[a] */
                  
/*     // load the knot ua */
/*     if (a != p) */
/*       for (i = 0; i < ph-oldr; i++) */
/*       { */
/*         newU[kind] = ua; */
/*         kind++; */
/*       } */

/*     // load ctrl pts into ic */
/*     for (j = lbz; j <= rbz; j++) */
/*     { */
/*       for (ii = 0; ii < nd; ii++) */
/*         newP[ii][cind] = ebpts[ii][j]; */
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
/*           bpts[ii][j] = P[ii][b-p+j]; */
/*       a = b; */
/*       b++; */
/*       ua = ub; */
/*     } */
/*     else */
/*       // end knot */
/*       for (i = 0; i <= ph; i++) */
/*         newU[kind+i] = ub; */
/*   }                   */
/*   // end while loop    */
  
/*   *nh = mh - ph - 1; */

/*   freematrix(bezalfa); */
/*   freematrix(bpts); */
/*   freematrix(ebpts); */
/*   freematrix(Nextbpts); */
/*   free(alfa); */
/* } */



/* curve_global_interp_mat */
/*
Compute the global curve interpolation matrix. 

Input:

- p: degree of the B-spline
- Q: points through which the curve should pass (nc,nd)
- nc: number of points = number of control points = n+1
- nd: dimension of the points (3 or 4)
- u: parameter values at the points (nc)
strategies:
  0 : equally spaced (not recommended)
  1 : chord length
  2 : centripetal (recommended)

Output:
- P: control points P(nc,nd)
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p
- A: coefficient matrix (nc,nc)

Modified algorithm A9.1 from 'The NURBS Book' pg369.
*/
  
static void curve_global_interp_mat(int p, double *Q, int nc, int nd, double *u, double *U, double *A)
{
  int n,m,i,j,s;

  n = nc - 1;
  m = nc + p;
  
  /* Compute the knot vector U by averaging (9.8) */
  for (i=0; i<m-p; ++i) U[i] = 0.0;
  for (i=m-p; i<=m; ++i) U[i] = 1.0;
  for (j=1; j<=n-p; ++j) {
    for (i=j; i<j+p; ++i) U[j+p] += u[i];
    U[j+p] /= p;
  }
  /* Set up coefficient matrix A */
  for (i=0; i<nc*nc; ++i) A[i] = 0.0;
  for (i=0; i<nc; ++i) {
    s = find_span(U,u[i],p,nc-1);
    basis_funs(U,u[i],p,s,A+i*nc+s-p); /* i-th row */
  }
}


/********************************************************/
/*********************** SURFACE ************************/
/********************************************************/



/* surface_points */
/*
Compute points on a B-spline surface. 

Input:

- P: control points P(ns,nt,nd)
- ns,nt: number of control points 
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]
- nU: number of knot values U = m+1
- V: knot sequence: V[0] .. V[n]
- nV: number of knot values V = n+1
- u: parametric values (nu,2): U[0] <= ui[0] <= U[m], V[0] <= ui[1] <= V[m]
- nu: number of parametric values

Output:
- pnt: (nu,nd) points on the B-spline

Modified algorithm A3.5 from 'The NURBS Book' pg103.
*/
static void surface_points(double *P, int ns, int nt, int nd, double *U, int nU, double *V, int nV, double *u, int nu, double *pnt)
{
  int i, j, k, p, q, su, sv, iu, iv;
  double S;
  
  /* degrees of the spline */
  p = nU - ns - 1;
  q = nV - nt - 1;

  /* space for the basis functions */
  double *Nu = (double*) malloc((p+1)*sizeof(double));
  double *Nv = (double*) malloc((q+1)*sizeof(double));

  /* for each parametric point j */
  for (j=0; j<nu; ++j) {

    /* find the span index of u[j] */
    su = find_span(U,u[2*j],p,ns-1);
    basis_funs(U,u[2*j],p,su,Nu);

    /* find the span index of v[j] */
    sv = find_span(V,u[2*j+1],q,nt-1);
    basis_funs(V,u[2*j+1],q,sv,Nv);

    iu = su-p;
    iv = sv-q;
    for (i=0; i<nd; ++i) {
      S = 0.0;
      for (k=0; k<=p; ++k) {
	S += Nu[k] * dotprod(Nv,1,P+((iu+k)*nt+iv)*nd+i,nd,q+1);
      }
      pnt[j*nd+i] = S;
    }
  }
  free(Nu);
  free(Nv);
}


/* surface_derivs */
/*
Compute derivatives of a B-spline surface. 

Input:

- mu,mv: number of derivatives to compute in u,v direction
- P: control points P(ns,nt,nd)
- ns,nt: number of control points
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]
- nU: number of knot values U = m+1
- V: knot sequence: V[0] .. V[n]
- nV: number of knot values V = n+1
- u: parametric values (nu,2): U[0] <= ui[0] <= U[m], V[0] <= ui[1] <= V[m]
- nu: number of parametric values

Output:
- pnt: (mu+1,mv+1,nu,nd) points and derivatives on the B-spline surface

Modified algorithm A3.6 from 'The NURBS Book' pg111.
*/
static void surface_derivs(int mu, int mv, double *P,int ns, int nt, int nd, double *U, int nU, double *V, int nV, double *u, int nu, double *pnt)
{
  int p,q,du,dv,su,sv,i,j,k,l,iu,iv,r;
  double S, *qnt;
  
  /* degrees of the spline */
  p = nU - ns - 1;
  q = nV - nt - 1;

  /* number of nonzero derivatives to compute */
  du = min(p,mu);
  dv = min(q,mv);

  /* space for the basis functions and derivatives */
  double *Nu = (double*) malloc((du+1)*(p+1)*sizeof(double));
  double *Nv = (double*) malloc((dv+1)*(q+1)*sizeof(double));

  /* clear everything */
  for (i=0; i<(mu+1)*(mv+1)*nu*nd; ++i) pnt[i] = 0;

  /* for each parametric point j */
  for (j=0; j<nu; ++j) {

    /* find the span index of u[j] */
    su = find_span(U,u[2*j],p,ns-1);
    basis_derivs(U,u[2*j],p,su,du,Nu);

    /* find the span index of v[j] */
    sv = find_span(V,u[2*j+1],q,nt-1);
    basis_derivs(V,u[2*j+1],q,sv,dv,Nv);

    /* for each nonzero dervative */
    for (k=0; k<=du; ++k) {
      for (l=0; l<=dv; ++l) {
	qnt = pnt + (k*(mv+1) + l) *nu*nd;

	iu = su-p;
	iv = sv-q;
	for (i=0; i<nd; ++i) {
	  S = 0.0;
	  for (r=0; r<=p; ++r) {
	    S += Nu[r] * dotprod(Nv,1,P+((iu+r)*nt+iv)*nd+i,nd,q+1);
	  }
	  qnt[j*nd+i] = S;
	}
      }      
    }
  }
  free(Nu);
  free(Nv);
}



/* surfaceDecompose */
/*
Decompose a Nurbs surface in Bezier patches. 

Input:

- p: degree of the B-spline
- P: control points P(nc,nd)
- nc: number of control points = n+1
- nd: dimension of the points (3 or 4)
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p

Output:

- newP: (nb*p+1,nd) new control points
- nb: number of Bezier segments 

Modified algorithm A5.7 from 'The NURBS Book' pg177.
*/
  

/* static void surfaceDecompose(double *P, int nc, int nd, double *U, int nk, double *newP) */
/* { */
/*   int i, j, k, p, s, m, r, a, b, mult, n, nb, ii, save; */
/*   double numer, alpha, *alfa; */

/*   n = nc - 1; */
/*   m = nk - 1; */
/*   p = m - n - 1; */

/*   alfa = (double *) malloc(p*sizeof(double)); */

/*   a = p; */
/*   b = p+1; */
/*   nb = 0; */
  
/*   /\* First bezier segment *\/ */
/*   for (i = 0; i < (p+1)*nd; i++) newP[i] = P[i]; */

/*   // Loop through knot vector *\/ */
/*   while (b < m) { */
/*     i = b; */
/*     while (b < m && U[b] == U[b+1]) b++; */
/*     mult = b-i+1; */
    
/*     if (mult < p) { */
/*       printf("mult at %d is %d < %d\n",b,mult,p); */
/*       /\* compute alfas *\/ */
/*       numer = U[b] - U[a]; */
/*       for (k = p; k > mult; k--) */
/*         alfa[k-mult-1] = numer / (U[a+k]-U[a]); */

/*       /\* Insert knot U[b] r times *\/ */
/*       r = p - mult; */
/*       for (j = 1; j <= r; j++) { */
/*         save = r - j; */
/*         s = mult + j; 	/\* Number of new points *\/ */
/*         for (k = p; k >= s; k--) { */
/* 	  alpha = alfa[k-s]; */
/* 	  printf("alpha = %f\n",alpha); */
/*           for (ii = 0; ii < nd; ii++) { */
/*             newP[(nb+k)*nd+ii] = alpha*newP[(nb+k)*nd+ii] + (1.0-alpha)*newP[(nb+k-1)*nd+ii]; */
/* 	    printf("Setting element %d to %f\n",(nb+k)*nd+ii,newP[(nb+k)*nd+ii]); */
/* 	  } */
/* 	} */
/* 	if (b < m) */
/* 	  /\* Control point of next segment *\/ */
/* 	  for (ii = 0; ii < nd; ii++) { */
/* 	    newP[(nb+p+save)*nd+ii] = newP[(nb+p)*nd+ii]; */
/* 	    printf("Copying element %d to %f\n",(nb+p+save)*nd+ii,newP[(nb+p+save)*nd+ii]); */
/* 	  } */
/*       } */
/*     } */
/*     /\* Bezier segment completed *\/ */
/*     nb += p; */
/*     if (b < m) { */
/*       /\* Initialize for next segment *\/ */
/*       for (i = r; i <= p; i++) */
/*         for (ii = 0; ii < nd; ii++) { */
/*           newP[(nb+i)*nd+ii] = P[(b-p+i)*nd+ii]; */
/* 	  printf("Initializing element %d to %f\n",(nb+i)*nd+ii,newP[(nb+i)*nd+ii]); */
/* 	} */
/*       a = b; */
/*       b++; */
/*     } */
/*   } */
  
/*   free(alfa); */
/* } */




/********************************************************/
/****** EXPORTED FUNCTIONS (callable from Python ********/
/********************************************************/



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

static PyObject * binomial(PyObject *self, PyObject *args)
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

static PyObject * allBernstein(PyObject *self, PyObject *args)
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
  all_bernstein(n,u,B);

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

static PyObject * curvePoints(PyObject *self, PyObject *args)
{
  int nd, nc, nk, nu;
  npy_intp *P_dim, *U_dim, *u_dim, dim[2];
  double *P, *U, *u, *pnt;
  PyObject *a1, *a2, *a3;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret=NULL;

  if (!PyArg_ParseTuple(args, "OOO", &a1, &a2, &a3))
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

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  nc = P_dim[0];
  nd = P_dim[1];
  nk = U_dim[0];
  nu = u_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);

  /* Create the return array */
  dim[0] = nu;
  dim[1] = nd;
  ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  pnt = (double *)PyArray_DATA(ret);

  /* Compute */
  curve_points(P, nc, nd, U, nk, u, nu, pnt);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  return ret;

 fail:
  //printf("error cleanup and return\n");
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

static PyObject * curveDerivs(PyObject *self, PyObject *args)
{
  int nc, nd, nk, nu, n;
  npy_intp *P_dim, *U_dim, *u_dim, dim[3];
  double *P, *U, *u, *pnt;
  PyObject *a1, *a2, *a3;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret=NULL;

  if(!PyArg_ParseTuple(args, "OOOi", &a1, &a2, &a3, &n))
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

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  nc = P_dim[0];
  nd = P_dim[1];
  nk = U_dim[0];
  nu = u_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);

  /* Create the return array */
  dim[0] = n+1;
  dim[1] = nu;
  dim[2] = nd;
  ret = PyArray_SimpleNew(3,dim, NPY_DOUBLE);
  pnt = (double *)PyArray_DATA(ret);

  /* Compute */
  curve_derivs(n, P, nc, nd, U, nk, u, nu, pnt);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  return ret;

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  return NULL;
}


static char curveKnotRefine_doc[] =
"Refine curve knot vector.\n\
\n\
Input:\n\
\n\
- P: control points P(nc,nd)\n\
- U: knot sequence: U(nk) (nk = nc+p+1)\n\
- u: (nu) parametric values of new knots: U[0] <= u[i] <= U[m]\n\
\n\
Output:\n\
- newP: (nc+nu,nd) new control points\n\
- newU: (m+nu) new knot vector\n\
\n\
Modified algorithm A5.1 from 'The NURBS Book' pg164.\n\
\n";

static PyObject * curveKnotRefine(PyObject *self, PyObject *args)
{
  int nd, nc, nk, nu;
  npy_intp *P_dim, *U_dim, *u_dim, dim[2];
  double *P, *U, *u, *newP, *newU;
  PyObject *a1, *a2, *a3;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret1=NULL, *ret2=NULL;

  if(!PyArg_ParseTuple(args, "OOO", &a1, &a2, &a3))
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

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  nc = P_dim[0];
  nd = P_dim[1];
  nk = U_dim[0];
  nu = u_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);

  /* Create the return arrays */
  dim[0] = nc+nu;
  dim[1] = nd;
  ret1 = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  newP = (double *)PyArray_DATA(ret1);
  dim[0] = nk+nu;
  ret2 = PyArray_SimpleNew(1,dim, NPY_DOUBLE);
  newU = (double *)PyArray_DATA(ret2);

  /* Compute */
  curve_knot_refine(P, nc, nd, U, nk, u, nu, newP, newU);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  //return ret1;
  return Py_BuildValue("(OO)", ret1, ret2);

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  return NULL;
}


static char curveDecompose_doc[] =
"Decompose a Nurbs curve in Bezier segments.\n\
\n\
Input:\n\
\n\
- P: control points P(nc,nd)\n\
- nc: number of control points = n+1\n\
- nd: dimension of the points (3 or 4)\n\
- U: knot sequence U(nk) with nk = nc+p+1 = nc+p\n\
\n\
Returns:\n\
\n\
- newP: (nb*p+1,nd) new control points defining nb Bezier segments\n\
\n\
Modified algorithm A5.6 from 'The NURBS Book' pg173.\n\
\n";

static PyObject * curveDecompose(PyObject *self, PyObject *args)
{
  int nd, nc, nk;
  npy_intp *P_dim, *U_dim, dim[2];
  double *P, *U, *newP;
  PyObject *a1, *a2;
  PyObject *arr1=NULL, *arr2=NULL, *ret=NULL;

  if(!PyArg_ParseTuple(args, "OO", &a1, &a2))
    return NULL;
  arr1 = PyArray_FROM_OTF(a1, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr1 == NULL)
    return NULL;
  arr2 = PyArray_FROM_OTF(a2, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr2 == NULL)
    goto fail;

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  nc = P_dim[0];
  nd = P_dim[1];
  nk = U_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);

  /* Compute number of knots to insert */
  int count = 0;
  int m = nk - 1;
  int p = nk - nc - 1;
  int b = p + 1;
  int i,mult;
  //printf("nc, nk, n, m, p = %d, %d, %d, %d, %d\n",nc,nk,nc-1,m,p);
  while (b < m) {
    i = b;
    while (b < m && U[b] == U[b+1]) b++;
    mult = b-i+1;
    //printf("b, i, mult = %d, %d, %d\n",b,i,mult);
    if (mult < p) {
      count += (p-mult);
      //printf("Count: %d\n",count);
    }
    b++;
  }
 
  /* Create the return arrays */
  dim[0] = nc+count;
  dim[1] = nd;
  ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  newP = (double *)PyArray_DATA(ret);

  /* Compute */
  curve_decompose(P, nc, nd, U, nk, newP);
  print_mat(newP,nc+count,nd);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  return ret;

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}

static char curveKnotRemove_doc[] =
"Refine curve knot vector.\n\
\n\
Input:\n\
\n\
- P: control points P(nc,nd)\n\
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p\n\
- u: knot value to remove: U[0] <= u <= U[m]\n\
- num: number of times to remove u\n\
- tol: allowable tolerance for deviation of the curve. See NURBS book, p. 185\n\
\n\
Output:\n\
- t: actual number of times that u was removed\n\
P and U are replaced with the new control points and knot vector\n\
\n\
Modified algorithm A5.8 from 'The NURBS Book' pg185.\n\
\n";

static PyObject * curveKnotRemove(PyObject *self, PyObject *args)
{
  int nd, nc, nk, num, t, i;
  npy_intp *P_dim, *U_dim, dim[2];
  double *P, *U, u, tol, *newP;
  PyObject *a1, *a2;
  PyObject *arr1=NULL, *arr2=NULL, *ret=NULL;

  if(!PyArg_ParseTuple(args, "OOfif", &a1, &a2, &u, &num, &tol))
    return NULL;
  arr1 = PyArray_FROM_OTF(a1, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr1 == NULL)
    return NULL;
  arr2 = PyArray_FROM_OTF(a2, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr2 == NULL)
    goto fail;

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  nc = P_dim[0];
  nd = P_dim[1];
  nk = U_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);

  /* Compute */
  t = curve_knot_remove(P, nc, nd, U, nk, u, num, tol);
  print_mat(P,nc-t,nd);
 
  /* Create the return arrays */
  dim[0] = nc-t;
  dim[1] = nd;
  ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  newP = (double *)PyArray_DATA(ret);
  for (i=0; i<dim[0]*dim[1]; ++i) newP[i] = P[i];

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  return ret;

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}


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

static char curveGlobalInterpolationMatrix_doc[] =
"Compute the global curve interpolation matrix.\n\
\n\
Input:\n\
\n\
- Q: points through which the curve should pass (nc,nd), where\n\
  nc is the number of points = number of control points = n+1 and\n\
  nd is the dimension of the points (3 or 4)\n\
- u: parameter values at the points (nc)\n\
- p: degree of the B-spline\n\
strategies:\n\
  0 : equally spaced (not recommended)\n\
  1 : chord length\n\
  2 : centripetal (recommended)\n\
\n\
Output:\n\
- P: control points P(nc,nd)\n\
- U: knot sequence: U[0] .. U[m]   m = n+p+1 = nc+p\n\
- A: coefficient matrix (nc,nc)\n\
\n\
Modified algorithm A9.1 from 'The NURBS Book' pg369.\n\
\n";

static PyObject * curveGlobalInterpolationMatrix(PyObject *self, PyObject *args)
{
  int p,nc,nd,nu;
  npy_intp *Q_dim, *u_dim, dim[2];
  double *Q, *u, *U, *A;
  PyObject *a1, *a2;
  PyObject *arr1=NULL, *arr2=NULL, *ret1=NULL, *ret2=NULL;

  if (!PyArg_ParseTuple(args, "OOi", &a1, &a2, &p))
    return NULL;
  arr1 = PyArray_FROM_OTF(a1, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr1 == NULL)
    return NULL;
  arr2 = PyArray_FROM_OTF(a2, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr2 == NULL)
    goto fail;

  Q_dim = PyArray_DIMS(arr1);
  u_dim = PyArray_DIMS(arr2);
  nc = Q_dim[0];
  nd = Q_dim[1];
  nu = u_dim[0];
  if (nu != nc) goto fail;

  Q = (double *)PyArray_DATA(arr1);
  u = (double *)PyArray_DATA(arr2);

  /* Create the return arrays */
  dim[0] = nc+p+1;
  ret1 = PyArray_SimpleNew(1,dim, NPY_DOUBLE);
  U = (double *)PyArray_DATA(ret1);
  dim[0] = nc;
  dim[1] = nc;
  ret2 = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  A = (double *)PyArray_DATA(ret2);

  /* Compute */
  curve_global_interp_mat(p, Q, nc, nd, u, U, A);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  //return ret1;
  return Py_BuildValue("(OO)", ret1, ret2);

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}


static char surfacePoints_doc[] =
"Compute points on a B-spline surface.\n\
\n\
Input:\n\
\n\
- P: control points P(ns,nt,nd)\n\
- ns,nt: number of control points\n\
- nd: dimension of the points (3 or 4)\n\
- U: knot sequence: U[0] .. U[m]\n\
- nU: number of knot values U = m+1\n\
- V: knot sequence: V[0] .. V[n]\n\
- nV: number of knot values V = n+1\n\
- u: parametric values (nu,2): U[0] <= ui[0] <= U[m], V[0] <= ui[1] <= V[m]\n\
- nu: number of parametric values\n\
\n\
Output:\n\
- pnt: (nu,nd) points on the B-spline\n\
\n\
Modified algorithm A3.5 from 'The NURBS Book' pg103.\n\
\n";


static PyObject * surfacePoints(PyObject *self, PyObject *args)
{
  int ns,nt,nd,nU,nV,nu;
  npy_intp *P_dim, *U_dim, *V_dim, *u_dim, dim[2];
  double *P, *U, *V, *u, *pnt;
  PyObject *a1, *a2, *a3, *a4;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *arr4=NULL, *ret=NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &a1, &a2, &a3, &a4))
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
  arr4 = PyArray_FROM_OTF(a4, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr4 == NULL)
    goto fail;

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  V_dim = PyArray_DIMS(arr3);
  u_dim = PyArray_DIMS(arr4);
  ns = P_dim[0];
  nt = P_dim[1];
  nd = P_dim[2];
  nU = U_dim[0];
  nV = V_dim[0];
  nu = u_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);
  V = (double *)PyArray_DATA(arr3);
  u = (double *)PyArray_DATA(arr4);

  /* Create the return array */
  dim[0] = nu;
  dim[1] = nd;
  ret = PyArray_SimpleNew(2,dim, NPY_DOUBLE);
  pnt = (double *)PyArray_DATA(ret);

  /* Compute */
  surface_points(P,ns,nt,nd,U,nU,V,nV,u,nu,pnt);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  Py_DECREF(arr4);
  return ret;

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);
  return NULL;
}


static char surfaceDerivs_doc[] =
"Compute derivatives of a B-spline surface.\n\
\n\
Input:\n\
\n\
- n: number of derivatives to compute\n\
- P: control points P(ns,nt,nd)\n\
- ns,nt: number of control points\n\
- nd: dimension of the points (3 or 4)\n\
- U: knot sequence: U[0] .. U[m]\n\
- nU: number of knot values U = m+1\n\
- V: knot sequence: V[0] .. V[n]\n\
- nV: number of knot values V = n+1\n\
- u: parametric values (nu,2): U[0] <= ui[0] <= U[m], V[0] <= ui[1] <= V[m]\n\
- nu: number of parametric values\n\
\n\
Output:\n\
- pnt: (n+1,nu,nd) points and derivatives on the B-spline surface\n\
\n\
Modified algorithm A3.6 from 'The NURBS Book' pg111.\n\
\n";

static PyObject * surfaceDerivs(PyObject *self, PyObject *args)
{
  int nc, nd, nk, nu, n;
  npy_intp *P_dim, *U_dim, *u_dim, dim[3];
  double *P, *U, *u, *pnt;
  PyObject *a1, *a2, *a3;
  PyObject *arr1=NULL, *arr2=NULL, *arr3=NULL, *ret=NULL;

  if(!PyArg_ParseTuple(args, "OOOi", &a1, &a2, &a3, &n))
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

  P_dim = PyArray_DIMS(arr1);
  U_dim = PyArray_DIMS(arr2);
  u_dim = PyArray_DIMS(arr3);
  nc = P_dim[0];
  nd = P_dim[1];
  nk = U_dim[0];
  nu = u_dim[0];
  P = (double *)PyArray_DATA(arr1);
  U = (double *)PyArray_DATA(arr2);
  u = (double *)PyArray_DATA(arr3);

  /* Create the return array */
  dim[0] = n+1;
  dim[1] = nu;
  dim[2] = nd;
  ret = PyArray_SimpleNew(3,dim, NPY_DOUBLE);
  pnt = (double *)PyArray_DATA(ret);

  /* Compute */
  curve_derivs(n, P, nc, nd, U, nk, u, nu, pnt);

  /* Clean up and return */
  Py_DECREF(arr1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  return ret;

 fail:
  //printf("error cleanup and return\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  return NULL;
}


static PyMethodDef _methods_[] =
{
	{"binomial", binomial, METH_VARARGS, binomial_doc},
	{"allBernstein", allBernstein, METH_VARARGS, allBernstein_doc},
	{"curvePoints", curvePoints, METH_VARARGS, curvePoints_doc},
	{"curveDerivs", curveDerivs, METH_VARARGS, curveDerivs_doc},
	{"curveKnotRefine", curveKnotRefine, METH_VARARGS, curveKnotRefine_doc},
	{"curveDecompose", curveDecompose, METH_VARARGS, curveDecompose_doc},
	{"curveKnotRemove", curveKnotRemove, METH_VARARGS, curveKnotRemove_doc},
	/* {"bspdegelev", nurbs_bspdegelev, METH_VARARGS, bspdegelev_doc}, */
	{"curveGlobalInterpolationMatrix", curveGlobalInterpolationMatrix, METH_VARARGS, curveGlobalInterpolationMatrix_doc},
	{"surfacePoints", surfacePoints, METH_VARARGS, surfacePoints_doc},
	{NULL, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initnurbs_(void)
{
  PyObject* module;
  module = Py_InitModule3("nurbs_", _methods_, __doc__);
  PyModule_AddIntConstant(module,"accelerated",1);
  import_array(); /* Get access to numpy array API */
}

/* End */

	
