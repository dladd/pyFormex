/*
  $Id$ 

  Scanner for ABAQUS .fil postprcessing files
  (C) 2008 Benedict Verhegghe
  Distributed under the GNU GPL version 3 or higher
  THIS PROGRAM COMES WITHOUT ANY WARRANTY
*/

#include <stdio.h>
#include <stdlib.h>


FILE * fil;
int lead;
long nw;
double * dp;
long * ip;
char * cp;

#define BUFSIZE 513
  
  
/* Process a single file */ 
int postprocess(const char* fn) {
  printf("Processing file '%s'\n",fn);
  fil = fopen(fn,"r");
  if (fil == NULL) return 1;
  if (fread(&lead,sizeof(lead),1,fil) != 1) return 2;
  printf("lead: %d\n",lead);

  int recnr = 0;
  while (!feof(fil)) {
    recnr++;
    if (fread(&nw,sizeof(nw),1,fil) != 1) return 3;
    printf("Record %d Length %d\n",recnr,nw);
    nw--;
    if (fread(dp,sizeof(*dp),nw,fil) != nw) return 4;
    printf("Record %d Length %d Type %d\n",recnr,nw,*ip);
  }
  fclose (fil);
  return 0;
}


int lead,tail;
union {
  double d[512];
  long   i[512];
  char   c[4096];
} data;

/* Process a single file */ 
int process(const char* fn) {
  printf("Processing file '%s'\n",fn);
  fil = fopen(fn,"r");
  if (fil == NULL) return 1;

  int recnr = 0;
  int blknr = 0;
  int i,nw,key;
  while (!feof(fil)) {
    blknr++;
    fread(&lead,sizeof(lead),1,fil);
    fread(&data,sizeof(data),1,fil);
    fread(&tail,sizeof(tail),1,fil);
    printf("** Block %d size %d lead %d tail %d\n",blknr,sizeof(data),lead,tail);
    for (i=0; i<512; ) {
      nw = data.i[i];
      key = data.i[i+1];
      recnr++;
      printf("Record %d Length %d Type %d\n",recnr,nw,key);
      if (nw <= 0 || key == 2001 || key == 0) break;
      i += nw;
    }
  }
  fclose (fil);
  return 0;
}

/* The main program llops over the files specified in the command line */
int main(int argc, char *argv[]) {  
  int i,nerr,res;
  
  printf("postabq 0.1 (C) 2008 Benedict Verhegghe\n");

  /* Allocate the buffer */
  dp = malloc(BUFSIZE * sizeof (*dp));
  if (dp == NULL) {
    printf("Could not allocate memory\n");
    return -1;
  }
  ip = (long*) dp;
  cp = (char*) dp;

  /* Loop over arguments */
  nerr = 0;
  for (i=1; i<argc; i++) {
    res = process(argv[i]);
    if (res != 0) {
      printf("ERROR %d\n",res);
      nerr++;
    }
  }

  /* Cleanup */
  free(dp);
  printf("Processed %d files, %d errors\n",i-1,nerr);
  return 0;
}

/* End */
