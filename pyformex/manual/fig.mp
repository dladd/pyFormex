verbatimtex
\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{eulervm}
\usepackage{bvmacros}
\usepackage{units}
\begin{document}
etex;

input mac_base;
input mac_label;
input macrosbalk;
input mac_rvw;
input graph;


  
beginfig(formex);
  a := 60u;
  b := 40u;
  z0 = (0,0);
  z1 = (a,0);
  z2 = (0,b);
  z3 = (0.8a,0.8b);
  pickup thickpen;
  arrowlabel.rt (btex axis 2: 3 coordinates$ etex, z0--z1);
  arrowlabel.bot (btex axis 1: \texttt{nplex} points$ etex, z0--z2);
  arrowlabel.lrt (btex axis 0: \texttt{nelems} elements$ etex, z0--z3);
endfig;
