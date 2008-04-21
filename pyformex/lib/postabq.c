/*
  $Id$ 

  Scanner for ABAQUS .fil results files
  (C) 2008 Benedict Verhegghe
  Distributed under the GNU GPL version 3 or higher
  THIS PROGRAM COMES WITHOUT ANY WARRANTY
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


char* copyright = "postabq 0.1 (C) 2008 Benedict Verhegghe";

FILE * fil;

/* Blocks and records
  A block consists of :
  - lead : 4 byte word with value 4096  (RECSIZE in bytes)
  - data : RECSIZE double words (512 * 8 = 4096 bytes)
  - tail : as lead

  A record consists of 
  - NW (1)  : number of (double) words
  - KEY (1) : record type
  - DATA (NW-2) : the data

  !!! Records may span the block boundary !!!
  Reading from file is done block by block. If we want to process records as
  a whole, we need to buffer at least 2 blocks.
*/

#define RECSIZE 512
#define BUFSIZE 2*RECSIZE
  
int recnr = 0;
int blknr = 0;
int err = 0;

int32_t lead,tail;
union {
  double d[BUFSIZE];
  int64_t i[BUFSIZE];
  char   c[8*BUFSIZE];
} data;

int64_t nw,key;

int64_t
  j,    /* Pointer to current data */
  jend, /* Pointer behind currently record */
  jmax; /* Pointer behind currently filled buffer */

#define STRINGBUFSIZE 256
char s[STRINGBUFSIZE];

int explicit = 0;  /* assume standard unless specified/detected */
int verbose = 0; 
int fake = 0; 

char* stripn(int64_t k,int64_t n,int strip) {
  s[0] = '\0';
  int64_t m = 8*n;
  if (m > STRINGBUFSIZE) m = STRINGBUFSIZE;
  memmove(s,data.c + (8*k),m);
  while (s[--m] == ' ') {}
  s[++m] = '\0';
  char* p = s;
  if (strip) {
    while (*p==' ') p++;
  }
  return p;
}

char* strn(int64_t k,int64_t n) {
  return stripn(k,n,0);
}

char* str(int64_t k) {
  return strn(k,1);
}

void do_element() {
  printf("D.Element(%d,",data.i[j++]);
  printf("'%s',[",str(j++));
  while (j < jend) printf("%d,",data.i[j++]);
  printf("])\n");
}

void do_node() {
  printf("D.Node(%d,[",data.i[j++]);
  int64_t j3 = j+3;
  if (j3 > jend) j3 = jend;
  while (j < j3) printf("%e,",data.d[j++]);
  if (j < jend) {
    printf("],normal=[");
    while (j < jend) printf("%e,",data.d[j++]);
  }
  printf("])\n");
}

void do_dofs() {
  printf("D.Dofs([");
  while (j < jend) printf("%d,",data.i[j++]);
  printf("])\n");
}

void do_outreq() {
  int64_t * ip = data.i + j++;
  int flag = ip[0];
  printf("D.OutputRequest(flag=%d,set='%s'",flag,str(j++));
  if (flag==0) printf(",eltyp='%s',",str(j++));
  printf(")\n");
}

void do_abqver() {
  printf("D.Abqver('%s')\n",str(j++));
  /* BEWARE ! Do not call str() multiple times in the same output instruction */
  printf("D.Date('%s',",strn(j,2)); j += 2;
  printf("'%s')\n",str(j++));
  printf("D.Size(nelems=%lld,nnodes=%lld,length=%f)\n",data.i[j],data.i[j+1],data.d[j+2]);
} 

void do_heading() {
  printf("D.Heading('%s')\n",strn(j,jend-j));
}

void do_nodeset() {
  printf("D.Nodeset('%s',[",stripn(j++,1,1));
  while (j<jend) printf("%d,",data.i[j++]);
  printf("])\n");
}

void add_nodeset() {
  printf("D.NodesetAdd([");
  while (j<jend) printf("%d,",data.i[j++]);
  printf("])\n");
}

void do_elemset() {
  printf("D.Elemset('%s',[",stripn(j++,1,1));
  while (j<jend) printf("%d,",data.i[j++]);
  printf("])\n");
}

void add_elemset() {
  printf("D.ElemsetAdd([");
  while (j<jend) printf("%d,",data.i[j++]);
  printf("])\n");
}

void do_label() {
  printf("D.Label(tag='%d',value='",data.i[j++]);
  printf("%s",strn(j,jend-j));
  printf("')\n");
}

void do_increment() {
  int64_t * ip = data.i + j;
  double * dp = data.d + j;
  int64_t type = ip[4];
  explicit = (type==17 || type == 74);
  printf("D.Increment(");
  printf("step=%d,",ip[5]);
  printf("inc=%d,",ip[6]);
  printf("tottime=%e,",dp[0]);
  printf("steptime=%e,",dp[1]);
  printf("timeinc=%e,",dp[10]);
  printf("type=%d,",type);
  printf("heading='%s',",stripn(j+11,10,1));
  if (!explicit) {
    printf("maxcreep=%e,",dp[2]);
    printf("solamp=%e,",dp[3]);
    printf("linpert=%d,",ip[7]);
    printf("loadfactor=%e,",dp[8]);
    printf("frequency=%e,",dp[9]);
  }
  printf(")\n");
}

void end_increment() {
  printf("D.EndIncrement()\n");
}

char* output_location[] = { "gp", "ec", "en", "rb", "na", "el" };
  
void do_elemheader() {
  int64_t * ip = data.i + j;
  int loc = ip[3];
  printf("D.ElemHeader(loc='%s',",output_location[loc]);
  printf("i=%d,",ip[0]);
  if (loc==0)
    printf("gp=%d,",ip[1]);
  else if (loc==2)
    printf("np=%d,",ip[1]);
  else if (ip[1]!=0)
    printf("ip=%d,",ip[1]);
  if (ip[2]!=0)
    printf("sp=%d,",ip[2]);
  if (loc==3)
    printf("rb='%s',",stripn(j+4,1,1));
  printf("ndi=%d,",ip[5]);
  printf("nshr=%d,",ip[6]);
  printf("nsfc=%d,",ip[8]);
  if (explicit)
    printf("ndir=%ld,",ip[7]);
  printf(")\n");
}

void do_elemout(char* text) {
  printf("D.ElemOutput('%s',[",text);
  while (j < jend) printf("%e,",data.d[j++]);
  printf("])\n");
}

void do_nodeout(char* text) {
  printf("D.NodeOutput('%s',%d,[",text,data.i[j++]);
  while (j < jend) printf("%e,",data.d[j++]);
  printf("])\n");
}

void do_total_energies() {
  double * dp = data.d + j;
  printf("D.TotalEnergies(");
  printf("ALLKE=%f,",dp[0]);
  printf("ALLSE=%f,",dp[1]);
  printf("ALLWK=%f,",dp[2]);
  printf("ALLPD=%f,",dp[3]);
  printf("ALLCD=%f,",dp[4]);
  printf("ALLVD=%f,",dp[5]);
  printf("ALLAE=%f,",dp[7]);
  printf("ALLIE=%f,",dp[10]);
  printf("ETOTAL=%f,",dp[11]);
  printf("ALLFD=%f,",dp[12]);
  printf("ALLDMD=%f,",dp[16]);
  if (explicit) {
    printf("ALLDC=%f,",dp[8]);
    printf("ALLIHE=%f,",dp[16]);
    printf("ALLHF=%f,",dp[17]);
  } else {
    printf("ALLKL=%f,",dp[6]);
    printf("ALLQB=%f,",dp[8]);
    printf("ALLEE=%f,",dp[9]);
    printf("ALLJD=%f,",dp[13]);
    printf("ALLSD=%f,",dp[14]);
  }
  printf(")\n");
}

/* Process the data of a record */ 
int process_data() {
  /* nw and key have been set, j points to data*/
  if (verbose) fprintf(stderr,"Record %d Offset %lld Length %lld Type %lld End %lld max %lld\n",recnr,j,nw,key,jend,jmax);
  if (fake) return 0;
  switch(key) {
  case 1900: do_element(); break;
  case 1901: do_node(); break;
  case 1902: do_dofs();  break;
  case 1911: do_outreq(); break;
  case 1921: do_abqver(); break;
  case 1922: do_heading(); break;
  case 1931: do_nodeset(); break;
  case 1932: add_nodeset(); break;
  case 1933: do_elemset(); break;
  case 1934: add_elemset(); break;
  case 1940: do_label(); break;
  case 2000: do_increment(); break;
  case 2001: end_increment(); break;

  case 1:   do_elemheader(); break;
  case 11:  do_elemout("S"); break;
  case 12:  do_elemout("SINV"); break;
  case 13:  do_elemout("SF"); break;
  case 101: do_nodeout("U"); break;
  case 102: do_nodeout("V"); break;
  case 103: do_nodeout("A"); break;
  case 104: do_nodeout("RF"); break;
  case 105: do_nodeout("EPOT"); break;
  case 106: do_nodeout("CF"); break;
  case 107: do_nodeout("COORD"); break;
  case 108: do_nodeout("POR"); break;
  case 109: do_nodeout("RVF"); break;
  case 110: do_nodeout("RVT"); break;

  case 1999: do_total_energies(); break;
  default: printf("D.Unknown(%lld)\n",key);
  }
  return err;
}


/* read the next block from file */
int read_block() {
  if (j < jmax) {
    /* Move the remaining data to the start of the buffer */
    int64_t nm = jmax-j;
    if (verbose) fprintf(stderr,"Moving %lld words to start of buffer\n",nm);
    memmove(data.d,data.d+j,8*nm);
    j = 0;
    jmax = j+nm;
  } else {
    j = 0;
    jmax = 0;
  }
  blknr++;
  if (verbose)
    fprintf(stderr,"Reading block at filepos %d, %d\n",ftell(fil),feof(fil));
  if ( fread(&lead,sizeof(lead),1,fil) != 1 && !feof(fil) ||
       !feof(fil) && fread(data.d+jmax,RECSIZE*8,1,fil) != 1 ||
       !feof(fil) && fread(&tail,sizeof(tail),1,fil) != 1 ) {
    fprintf(stderr,"ERROR while reading block nr %d at filepos %d\n",blknr,ftell(fil));
    return 1;
  }
  if (feof(fil)) return 1;
  jmax += RECSIZE;
  if (verbose) {
    fprintf(stderr,"** Block %d size %d lead %d tail %d\n",blknr,8*RECSIZE,lead,tail);
    fprintf(stderr,"** Buffer Start %lld End %lld size %lld\n",j,jmax,jmax-j);
  }
  return 0;
}

/* Process a single file */ 
int process_file(const char* fn) {
  fprintf(stderr,"Processing file '%s'\n",fn);
  fil = fopen(fn,"r");
  if (fil == NULL) return 1;

  printf("#!/usr/bin/env pyformex\n");
  printf("# Created by %s\n",copyright);
  printf("from plugins.fe_post import FeResult\n");
  printf("D = FeResult()\n");
  j = jmax = 0; /* start with empty buffer */
  while (!feof(fil)) {
    if (read_block()) break;
    while (j < jmax) { /* we have data : process them */
      nw = data.i[j];
      if (nw <= 0) {
	/* this must be block padding */
	if (verbose) fprintf(stderr,"Skipping rest of block(padding)\n)");
	j = jmax;
	break;
      }
      if (j+nw > jmax) {
	/* record spans block boundary */
	if (verbose) fprintf(stderr,"Record exceeds block boundary\n");
	break;
      }
      jend = j+nw;
      if (jend > jmax) {
	fprintf(stderr,"ERROR: record seems to span more than 2 blocks\n");
	return 1;
      }
      key = data.i[j+1];
      recnr++;
      j += 2;
      if (process_data()) return 1;
      j = jend; /* in case the process_data did not process everything */
    }
  }
  printf("D.Export()\n");
  printf("# End\n");
  fclose(fil);
  return 0;
}

void print_copyright() {
  fprintf(stderr,"%s\n",copyright);
}

void print_usage() {
  fprintf(stderr,"\nUsage: postabq [options] output.fil\n\
Converts an ABAQUS output file (.fil) into a Python script.\n\
The output goes to stdout.\n\
\n\
Options:\n\
  -v : Be verbose (mostly for debugging)\n\
  -e : Force EXPLICIT from the start (default is to autodetect)\n\
  -n : Dry run: run through the file but do not produce conversion\n\
  -h : Print this help text\n\
  -V : Print version and exit\n\
\n");
}

/* The main program loops over the files specified in the command line */
int main(int argc, char *argv[]) {  
  int i,nerr,res,nfiles;
  char c;
  
  print_copyright();

  /* Process command line options */
  for (i=1; i<argc; i++) {
    if (argv[i][0] != '-') continue;
    c = argv[i][1];
    switch (c) {
    case 'v': verbose=1; break;
    case 'e': explicit=1; break;
    case 'n': fake=1; break;
    case 'h': print_usage();
    case 'V': return 0;
    default: fprintf(stderr,"Invalid option '%c'; use '-h' for help\n",c);
      return 1;
    }
  }

  /* Loop over non-option arguments */
  nerr = 0;
  nfiles = 0;
  for (i=1; i<argc; i++) {
    if (argv[i][0] == '-') continue;
    nfiles ++;
    res = process_file(argv[i]);
    if (res != 0) {
      fprintf(stderr,"ERROR %d\n",res);
      nerr++;
    }
  }

  /* Cleanup */
  fprintf(stderr,"Processed %d files, %d errors\n",nfiles,nerr);
  return 0;
}

/* End */
