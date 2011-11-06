#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex    
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""
sloc.py

Create a sloccount report for the pyformex tagged releases.
(C) Benedict Verhegghe
"""

print __doc__
from pyformex.flatkeydb import FlatDB
from pyformex.utils import runCommand,removeTree

import os,sys,commands,datetime

now = datetime.datetime.now().replace(microsecond=0).isoformat(' ')
print now

curdir = os.path.dirname(__file__)
dbfile = os.path.join(curdir,'pyformex-releases.fdb')
if not os.path.exists(dbfile):
    print("The dbfile %s does not exist"%dbfile)
    sys.exit()
    
DB = FlatDB(req_keys=['tag','date','rev'],beginrec = 'release',endrec = '')

DB.readFile(dbfile)
keys = DB.keys()
keys.sort()

print "List of releases: %s" % keys

tmpdir = '_sloccount_'
workdir = os.path.join(curdir,tmpdir)


def runCmd(cmd):
    print cmd
    return commands.getstatusoutput(cmd)


def sloccount(rel):
    """Create a sloccount report for release"""
    tag = rel['tag']
    rev = rel['rev']
    print "Processing release %s" % tag

    slocfile = "pyformex.sloc.%s" % tag
    if os.path.exists(slocfile):
        print "  %s exists: skipping" % slocfile
        return
        
    if not os.path.exists(workdir):
        cmd = "svn co svn://svn.berlios.de/pyformex/trunk -r%s %s" % (rev,tmpdir)
        runCmd(cmd)

    cmd = "cd %s;svn up -r%s" % (tmpdir,rev)
    runCmd(cmd)

    pyfdir = os.path.join(tmpdir,'pyformex')
    if not os.path.isdir(pyfdir):
        pyfdir = tmpdir
    print "SLOCCOUNTING %s" % pyfdir
    cmd = "sloccount %s > %s" % (pyfdir,slocfile)
    runCmd(cmd)

    


for release in keys:
    sloccount(DB[release])
    removeTree(workdir)
    

# Now, create some statistics

def extract(filename):
    res,out = runCmd("gawk -f slocstats.awk %s" % filename)
    if res:
        raise ValueError,"Error extracting data"
    return dict([ DB.splitKeyValue(line) for line in out.split('\n') ])

KEYS=set([])
for release in keys:
    rel = DB[release]
    tag = rel['tag']
    rev = rel['rev']
    slocfile = "pyformex.sloc.%s" % tag
    print "Processing %s" % slocfile
    rel.update(extract(slocfile))
    KEYS |= set(rel.keys())

# No, better use a standard order!
KEYSorted = "date rev tag size python ansic sh sloc manyears years dollars"
KEYS = KEYSorted.split()

DB.writeFile('pyformex-stats.db')

out = "#"+' '.join(KEYS)+'\n'

for release in keys:
    rel = DB[release]
    val = [ rel.get(k,'*') for k in KEYS ]
    out += ' '.join(val)+'\n'

print out
statsfile = file('pyformex-stats.dat','w')
statsfile.write(out)
statsfile.close()

outfile = 'pyformex-stats.png'

gnu = """set terminal png size 640,480
set output "%s"
set datafile missing '*'
set title "pyFormex history (http://pyformex.berlios.de)\\nCreated %s"
set key top left
#set offsets 0,0.1,0,0
set xdata time
set timefmt "%s"
set format x "%s"
set xlabel "Year/Month"
#set ylabel "revision number"
#set yrange [0:1.2]
plot """ % (outfile,now,'%Y-%m-%d','%y\\n%m')

KEYSplot = { # what to plot (scale, title)
    'python': (100,'%S lines of Python code'),
    'ansic': (10,'%S lines of C code'),
    'manyears': (0.01,'%S man-years'),
    'dollars': (1000,'%S dollars'),
    'rev': (None,'number of revisions'),
    }

for key in KEYSplot:
    col = KEYS.index(key) + 1
    scale,title = KEYSplot[key]
    if scale:
        item = "($%s/%s)" % (col,scale)
        title = title.replace('%S',str(scale))
    else:
        item = "%s" % col
    gnu += "\\\n  'pyformex-stats.dat' using 1:%s title '%s'," % (item,title)

gnu = gnu[:-1] + '\n'

gnufile = 'pyformex-test.gnu'

fil = open(gnufile,'w')
fil.write(gnu)
fil.close()

runCommand('gnuplot %s && display %s' % (gnufile,outfile))

# End
