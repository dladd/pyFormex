#!/usr/bin/env python
"""Translate formian code to python

No change :
  + - * / 
  sign (defined further)
  abs 
  sqrt,sin,cos,tan,asin,acos,atan,exp (from math)
  
  ln -> log (from math)
  ric -> int(round())
  tic -> int()
  floc -> float()
  m^n -> pow(m,n) of m**n
  f|x -> f(x)

  tran(i,j)|F -> F.translate(i-1,j)
  ref(i,j)|F  -> F.reflect(i-1,j)

"""


def sign(a):
    if a < 0:
        return -1
    else:
        return 1

