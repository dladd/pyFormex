/*
  $Id$ 

  Scanner for ABAQUS .fil postprcessing files
  (C) 2008 Benedict Verhegghe
  Distributed under the GNU GPL version 3 or higher
  THIS PROGRAM COMES WITHOUT ANY WARRANTY
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE * fil;

int lead,tail;
union {
  double d[512];
  long   i[512];
  char   c[4096];
} data;
int nw,key;

/* union { */
/*   double* d; */
/*   long*   i; */
/*   const char*   c; */
/* } dp; */

char s[256];

/* Print 8 characters starting from p */
void printc8(int j) {
  const char* p = data.c + 8*j;
  int i;
  for (i=0; i<8; ) putchar(p[i++]);
}

char * to_str(int j) {
  strncpy(s,data.c+j,8);
  s[8] = '\0';
  return s;
}

void do_element(int j) {
  int i;
  printf("Element %d ",data.i[j++]);
  printc8(j++);
  for (i=0; i<nw-4; i++) printf(" %d",data.i[j+i]);
  printf("\n");
}

void do_node(int j) {
  int i;
  printf("Node %d ",data.i[j++]);
  for (i=0; i<nw-3; i++) printf(" %e",data.d[j+i]);
  printf("\n");
}

void do_dofs(int j) {
  int i;
  printf("Dofs ");
  for (i=0; i<nw-2; i++) printf(" %d",data.i[j+i]);
  printf("\n");
}

void do_outreq(int j) {
  int i;
  printf("OutputRequest(");
  printf("typ=%d,",data.i[j++]);
  printf("set='%s',",to_str(j++));
  printf("eltyp='%s',",to_str(j++));
  printf(")\n");
}

void do_abqver(int j) {
  printf("Abqver ");
  printc8(j++);
  printc8(j++);
  printc8(j++);
  printc8(j++);
  printf("  %d elements, %d nodes  %f\n",data.i[j++],data.i[j++],data.d[j++]);
} 

void do_heading(int j) {
  int i;
  printf("Heading");
  for (i=0; i<nw-2; i++) printc8(j++);
  printf("\n");
}

void do_nodeset(int j) {
  int i;
  printf("Nodeset ");
  printc8(j++);
  for (i=0; i<nw-3; i++) printf(" %d",data.i[j+i]);
  printf("\n");
}

void do_elementset(int j) {
  int i;
  printf("Elementset ");
  printc8(j++);
  for (i=0; i<nw-3; i++) printf(" %d",data.i[j+i]);
  printf("\n");
}

void do_label(int j) {
  int i;
  printf("Label %d ",data.i[j++]);
  for (i=0; i<nw-3; i++) printc8(j++);
  printf("\n");
}

void do_increment(int j) {
  int i;
  printf("Increment ");
  for (i=0; i<4; ++i) printf(" %e",data.d[j++]);
  j++;
  for (i=0; i<3; ++i) printf(" %d",data.i[j++]);
  for (i=0; i<3; ++i) printf(" %e",data.d[j++]);
  printf("\n");
  printf("Stepheading ");
  for (i=0; i<10; i++) printc8(j++);
  printf("\n");
}

void do_total_energies(j) {
  printf("TotalEnergies(dict(");
  printf("ALLKE=%f,",data.d[j++]);
  printf("ALLSE=%f,",data.d[j++]);
  printf("ALLWK=%f,",data.d[j++]);
  printf("ALLPD=%f,",data.d[j++]);
  printf("ALLCD=%f,",data.d[j++]);
  printf("ALLVD=%f,",data.d[j++]);
  printf("ALLKL=%f,",data.d[j++]);
  printf("ALLAE=%f,",data.d[j++]);
  printf("ALLQB=%f,",data.d[j++]);
  printf("ALLEE=%f,",data.d[j++]);
  printf("ALLIE=%f,",data.d[j++]);
  printf("ETOTAL=%f,",data.d[j++]);
  printf("ALLFD=%f,",data.d[j++]);
  printf("ALLJD=%f,",data.d[j++]);
  printf("ALLSD=%f,",data.d[j++]);
  printf("ALLSD=%f,",data.d[j++]);
  printf("ALLDMD=%f,",data.d[j++]);
  printf("unused1=%f,",data.d[j++]);
  printf("unused2=%f,",data.d[j++]);
  printf("))\n");
}

/* Process a record */ 
int process_record(j) {
  switch(key) {
  case 1900: do_element(j); break;
  case 1901: do_node(j); break;
  case 1902: do_dofs(j);  break;
  case 1911: do_outreq(j); break;
  case 1921: do_abqver(j); break;
  case 1922: do_heading(j); break;
  case 1931: do_nodeset(j); break;
  case 1933: do_elementset(j); break;
  case 1940: do_label(j); break;
  case 2000: do_increment(j); break;
  case 1999: do_total_energies(j); break;
  default: printf("Unknown record type %d\n",key);
  }
  return 0;
}


/* Process a single file */ 
int process_file(const char* fn) {
  printf("Processing file '%s'\n",fn);
  fil = fopen(fn,"r");
  if (fil == NULL) return 1;

  int recnr = 0;
  int blknr = 0;
  int i;
  while (!feof(fil)) {
    blknr++;
    fread(&lead,sizeof(lead),1,fil);
    fread(&data,sizeof(data),1,fil);
    fread(&tail,sizeof(tail),1,fil);
/*     printf("** Block %d size %d lead %d tail %d\n",blknr,sizeof(data),lead,tail); */
    for (i=0; i<512; ) {
      nw = data.i[i];
      key = data.i[i+1];
      recnr++;
/*       printf("Record %d Length %d Type %d\n",recnr,nw,key); */
      if (nw <= 0 || key == 2001 || key == 0) break;
/*       dp.i = data.i + i + 2; */
      process_record(i+2);
      i += nw;
    }
  }
  fclose (fil);
  return 0;
}

/* The main program loops over the files specified in the command line */
int main(int argc, char *argv[]) {  
  int i,nerr,res;
  
  printf("postabq 0.1 (C) 2008 Benedict Verhegghe\n");

  /* Loop over arguments */
  nerr = 0;
  for (i=1; i<argc; i++) {
    res = process_file(argv[i]);
    if (res != 0) {
      printf("ERROR %d\n",res);
      nerr++;
    }
  }

  /* Cleanup */
  printf("Processed %d files, %d errors\n",i-1,nerr);
  return 0;
}

/* End */
