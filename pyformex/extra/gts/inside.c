// $Id$ 
//
//  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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

/*
 * 
 * Determine whether points are inside a given closed surface or not.
 *
 */

#include <stdlib.h>
#include <locale.h>
#include <string.h>
#include "config.h"
#ifdef HAVE_GETOPT_H
#  include <getopt.h>
#endif /* HAVE_GETOPT_H */
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif /* HAVE_UNISTD_H */
#include "gts.h"

/* inside - check if points are inside a surface */
int main (int argc, char * argv[])
{
  GtsSurface * s1;
  GNode * tree1;
  GtsPoint P;
  FILE * fptr;
  GtsFile * fp;
  int c = 0, cnt = 0;
  double x,y,z;
  gboolean verbose = FALSE;
  gchar * file1, * file2;
  gboolean is_open1, is_inside;

  if (!setlocale (LC_ALL, "POSIX"))
    g_warning ("cannot set locale to POSIX");

  /* parse options using getopt */
  while (c != EOF) {
#ifdef HAVE_GETOPT_LONG
    static struct option long_options[] = {
      {"help", no_argument, NULL, 'h'},
      {"verbose", no_argument, NULL, 'v'}
    };
    int option_index = 0;
    switch ((c = getopt_long (argc, argv, "shv", 
			      long_options, &option_index))) {
#else /* not HAVE_GETOPT_LONG */
    switch ((c = getopt (argc, argv, "shv"))) {
#endif /* not HAVE_GETOPT_LONG */
    case 'v': /* verbose */
      verbose = TRUE;
      break;
    case 'h': /* help */
      fprintf (stderr,
	"Usage: gtsinside [OPTION] FILE1 FILE2\n"
	"Test whether points are inside a closed surface.\n"
	"FILE1 is a surface file. FILE2 is a text file where each line\n"
	"contains the three coordinates of a point, separated with blanks.\n"
	"\n"
	"  -v      --verbose  print statistics about the surface\n"
	"  -h      --help     display this help and exit\n"
	"\n"
	"Reports bugs to %s\n",
	"https://savannah.nongnu.org/projects/pyformex/");
      return 0; /* success */
      break;
    case '?': /* wrong options */
      fprintf (stderr, "Try `gtsinside --help' for more information.\n");
      return 1; /* failure */
    }
  }

  if (optind >= argc) { /* missing FILE1 */
    fprintf (stderr, 
	     "gtsinside: missing FILE1\n"
	     "Try `inside --help' for more information.\n");
    return 1; /* failure */
  }
  file1 = argv[optind++];

  if (optind >= argc) { /* missing FILE2 */
    fprintf (stderr, 
	     "gtsinside: missing FILE2\n"
	     "Try `gtsinside --help' for more information.\n");
    return 1; /* failure */
  }
  file2 = argv[optind++];

  /* open first file */
  if ((fptr = fopen (file1, "rt")) == NULL) {
    fprintf (stderr, "gtsinside: can not open file `%s'\n", file1);
    return 1;
  }
  /* reads in first surface file */
  s1 = GTS_SURFACE (gts_object_new (GTS_OBJECT_CLASS (gts_surface_class ())));
  fp = gts_file_new (fptr);
  if (gts_surface_read (s1, fp)) {
    fprintf (stderr, "gtsinside: `%s' is not a valid GTS surface file\n", 
	     file1);
    fprintf (stderr, "%s:%d:%d: %s\n", file1, fp->line, fp->pos, fp->error);
    return 1;
  }
  gts_file_destroy (fp);
  fclose (fptr);

  /* open second file */
  if ((fptr = fopen (file2, "rt")) == NULL) {
    fprintf (stderr, "gtsinside: can not open file `%s'\n", file2);
    return 1;
  }

  /* display summary information about the surface */
  if (verbose) {
    gts_surface_print_stats (s1, stderr);
  }

  /* check that the surface is an orientable manifold */
  if (!gts_surface_is_orientable (s1)) {
    fprintf (stderr, "gtsinside: surface `%s' is not an orientable manifold\n",
  	     file1);
    return 1;
  }

  /* build bounding box tree for the surface */
  tree1 = gts_bb_tree_surface (s1);
  is_open1 = gts_surface_volume (s1) < 0. ? TRUE : FALSE;

  /* scan file 2 for points and determine if it they are inside surface */
  while ( fscanf(fptr, "%lg %lg %lg", &x, &y, &z) == 3 ) {
    P.x = x; P.y = y; P.z = z;
    is_inside = gts_point_is_inside_surface(&P,tree1,is_open1);
    if (is_inside) printf("%d\n",cnt);
    //printf("Point %d: %lf, %lf, %lf: %d\n",cnt,P.x,P.y,P.z,is_inside);
    cnt++;
  }
  if ( !feof(fptr) ) {
    fprintf (stderr, "gtsinside: error while reading points from file `%s'\n",
  	     file2);
    return 1;
  }
  fclose (fptr);

  /* destroy surface */
  gts_object_destroy (GTS_OBJECT (s1));

  /* destroy bounding box tree (including bounding boxes) */
  gts_bb_tree_destroy (tree1, TRUE);

  return 0;
}
