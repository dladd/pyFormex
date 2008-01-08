/* $Id: */

/*
  Misc methods.
*/

#include <Python.h>
#include <arrayobject.h>

/**************************************************** system ****/
/* Execute a system command */
static PyObject *
drawgl_system(PyObject *dummy, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return Py_BuildValue("i", sts);
}

/**************************************************** printf ****/
/* Print a string */
static PyObject *
drawgl_printf(PyObject *dummy, PyObject *args)
{
    const char *command;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    printf("COMMAND %s\n",command);
    return Py_BuildValue("s", command);
}



/**************************************************** isclose ****/
/* Check if two floats are equal within a given tolerance */
int isclose(float a, float b, float rtol, float atol)
{
  printf("Compare %e %e %e %e\n",a,b,rtol,atol); 
  int ok;
  ok = fabs(a-b) < atol + rtol * fabs(b);
  printf("  a-b: %e; atol+rtol*b %e\n",fabs(a-b),atol + rtol * fabs(b));
  printf("  ok: %d\n",ok);
  return ok;
}

/**************************************************** addar ****/
/* Add two arrays */
static PyObject *
drawgl_addar(PyObject *dummy, PyObject *args)
{
  PyObject *arg1=NULL, *arg2=NULL;//, *out=NULL;
  PyObject *arr1=NULL, *arr2=NULL;//, *oarr=NULL;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;
  printf("Parsed the arguments\n");
  arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  printf("Got argument 1\n");
  arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_INOUT_ARRAY);
  if (arr2 == NULL) goto fail;
  printf("Got argument 2\n");
/*   oarr = PyArray_FROM_OTF(out, NPY_FLOAT, NPY_INOUT_ARRAY); */
/*   if (oarr == NULL) goto fail; */
/*   printf("Got argument 3\n"); */
  /* code that makes use of arguments */
  /* We suppose the dimensions are correct*/
  int nd,i;
  npy_intp * dims;
  float *a=NULL, *b=NULL;//, *c=NULL;
  nd = PyArray_NDIM(arg1);
  printf("Arrays have %d dimensions\n",nd);
  dims = PyArray_DIMS(arg1);
  printf("Arrays have dimension %d x %d\n",(int)(dims[0]),(int)(dims[1]));
  a = (float *)PyArray_DATA(arr1);
  b = (float *)PyArray_DATA(arr2);
  for (i=0; i<dims[0]; i++) printf(" %f",a[i]);
  printf("\n");
  for (i=0; i<dims[0]; i++) printf(" %f",b[i]);
  printf("\n");
  for (i=0; i<dims[0]; i++) a[i] += b[i];
  for (i=0; i<dims[0]; i++) printf(" %f",a[i]);
  printf("\n");
  float v1=0.01,v2=0.010001,rtol=1.e-3,atol=1.e-5;
  printf("%f and %f are ",v1,v2);
  if (!isclose(v1,v2,rtol,atol)) printf(" NOT ");
  printf("close!\n");
  v2 = 0.1002;
  printf("%f and %f are ",v1,v2);
  if (!isclose(v1,v2,rtol,atol)) printf(" NOT ");
  printf("close!\n");
/*   c = (float *)PyArray_DATA(oarr); */
  
  /* Clean up and return */
  printf("Cleaning up\n");
  Py_DECREF(arr1);
  Py_DECREF(arr2);
/*   Py_DECREF(oarr); */
  Py_INCREF(Py_None);
  return Py_None;
 fail:
  printf("Error Cleanup\n");
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
/*   PyArray_XDECREF_ERR(oarr); */
  return NULL;
}


/**************************************************** coords.fuse ****/
/* Fuse nodes */
/* args:  x, val, flag, sel, tol
     x   : (nnod,3) coordinates
     val : (nnod) gives the point a code, such that only points with equal val
         are close. points in x are sorted according to val.
     flag: (nnod) initially 1, set to 0 if point is fused with another.
     sel : (nnod) 
     tol : tolerance for defining equality of coordinates
*/  
static PyObject *
coords_fuse(PyObject *dummy, PyObject *args)
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
/*   printf("Fast Fuse Data:\n"); */
/*   for (i=0; i<nnod; i++) { */
/*     ki = 3*i; */
/*     printf(" node %d: %12.8f, %12.8f, %12.8f; %lld; %d; %lld\n",i,x[ki],x[ki+1],x[ki+2],val[i],flag[i],sel[i]); */
/*   } */
  for (i=0; i<nnod; i++) {
    ki = 3*i;
/*     printf(" node %d: %12.8f, %12.8f, %12.8f; %lld; %d; %lld\n",i,x[ki],x[ki+1],x[ki+2],val[i],flag[i],sel[i]); */
  
    j = i-1;
    while (j >= 0 && val[i]==val[j]) {
/*       printf("Compare %d and %d\n",i,j); */
      kj = 3*j;
      if ( fabs(x[ki]-x[kj]) < tol && \
	   fabs(x[ki+1]-x[kj+1]) < tol && \
	   fabs(x[ki+2]-x[kj+2]) < tol ) {
/* 	printf("Nodes %d and %d are close\n",i,j);	\ */
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
/*   for (i=0; i<nnod; i++) { */
/*     ki = 3*i; */
/*     printf(" node %d: %12.8f, %12.8f, %12.8f; %ld; %d; %ld\n",i,x[ki],x[ki+1],x[ki+2],val[i],flag[i],sel[i]); */
/*   } */
  //printf("Cleaning up\n");
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
static PyMethodDef Methods[] = {
    {"system", drawgl_system, METH_VARARGS, "Execute a shell command."},
    {"printf", drawgl_printf, METH_VARARGS, "Print a string."},
    {"addar2", drawgl_addar, METH_VARARGS, "Add two arrays."},
    {"fuse", coords_fuse, METH_VARARGS, "Fuse nodes."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module */
PyMODINIT_FUNC
initmisc(void)
{
    (void) Py_InitModule("misc", Methods);
    import_array(); /* Get access to numpy array API */
}

/* End */
