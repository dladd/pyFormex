#!/usr/bin/gawk -f
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
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

#
# (c) 2009 Benedict Verhegghe
# This is an embryonal awk script to extract information from an Abaqus
# input file and write it in a format that can be loaded easily into
# pyFormex.
#
# Usage: gawk -f read_abq.awk ABAQUS_INPUT_FILE
#
# In most cases however, the user will call this command from the pyFormex
# GUI, using the Mesh plugin menu ("Convert Abaqus .inp file"), after which
# the model can be imported with the "Import converted model" menu item.
#
# Currently it reads nodes, elements and elsets and stores them in files
# partname.nodes, partname.elems and partname.esets
# There may be multiple nodes and elems files.
# An index file partname.mesh keeps record of the created files.
# nodes/elems defined before the first part get a default part name.
#  
# The nodes should be numbered consecutively in each of the parts,
# but not necessarily over the different parts.
#
#

######## scanner #######################

# initialisation: set default part name
BEGIN { mode=0; start_part("DEFAULT_PART"); }

# start a new part
/^\*\* PART INSTANCE:/ { start_part($4); print "**PART "$4; next; }
/^\*Part,/ { sub(".*name=",""); sub(" .*",""); start_part($0); print "*Part "$0; next; }

# skip all other comment lines
/^\*\*/ { next; }

# start a node block: record the number of the first node
/^\*Node/ { 
    start_mode(1)
    getline; gsub(",",""); header = "# nodes "outfile " offset "$1
}

# start an element block
/^\*Element,/ { 
    start_mode(2) 
    getline; header = "# elems "outfile " nplex "NF-1

}

# start an elset block
/^\*Elset, elset=.*, generate/ { 
    start_mode(3)
    sub(".*elset=","");sub(",.*",""); setname=$0;
    getline;
}

# skip other commands
/^\*/ { print "Unknown command: "$0;  end_mode(); next;}

# output data according to output mode
{ 
    if (mode==1) print_node();
    else if (mode==2) print_elem();    
    else if (mode==3) print_elset();    
}

END { end_mode(); fflush("") }

######## functions #####################

# start a new part with name pname
function start_part(pname) {
    partname = pname
    meshfile = partname".mesh"
    printf("") > meshfile
    nodesblk = -1
    elemsblk = -1
    esetsblk = -1
}


# start a new output file with given name and type
function start_mode(mod) {
    end_mode()

    mode = mod
    if (mode==1) {
	nodesblk = start_blocked_file("nodes",nodesblk)
    }
    else if (mode==2) {
	elemsblk = start_blocked_file("elems",elemsblk)
    }
    else if (mode==3) {
	esetsblk = start_unique_file("esets",esetsblk)
    }
    print "Starting mode "mode" to file "outfile 
    count = 0
}


# Start a file for a blocked type
function start_blocked_file(type,blk) {
    outfile = partname"."type
    blk += 1
    if (blk > 0) outfile = outfile""blk
    printf("") > outfile
    return (blk)
}

# Start a file for a unique type
function start_unique_file(type,blk) {
    outfile = partname"."type
    if (blk < 0) print "# "type" "outfile >> meshfile 
    printf("") > outfile
    return 0
}

# stop writing to the current file
function end_mode() {
    if (mode>0) {
	print "Ending mode "mode
	mode = 0
	if (count > 0) print header" count "count >> meshfile
    }
}

# print a node
function print_node() {
    gsub(",",""); print $2," ",$3," ",$4 >> outfile;
    count += 1;
}

# print an element
function print_elem() {
    gsub(",",""); print $2" "$3" "$4" "$5" "$6" "$7" "$8" "$9 >> outfile;
    count += 1;
}

# print an elset
function print_elset() {
    print setname
    print $0
    gsub(",",""); print setname" "$1" "$2" "$3 >> outfile;
}

# End