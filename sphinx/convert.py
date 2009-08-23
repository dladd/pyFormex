#!/usr/bin/env python
import sys
sys.path[0:0] = ['/home/bene/prj/doctools']
#print sys.path
from converter import restwriter, convert_file
     
for infile in sys.argv[1:]:
    outfile = infile.replace('.tex','.rst')
    print "%s -> %s" % (infile,outfile)
    try:
        convert_file(infile, outfile)
    except:
        raise
