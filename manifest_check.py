#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""manifest_check.py

Check that the files in MANIFEST are valid for distribution.
This is done separately from the MANIFEST creator, because manifest.py
is also included in the distribution, while this scipt is not.
"""
import pysvn
client = pysvn.Client()

def get_manifest_files():
    return [ f.strip('\n') for f in open('MANIFEST').readlines() if not f.startswith('#')]


check = {
    'M':[],
    'U':[],
    '?':[],
    '-':[],
}

    
def get_status(f):
    try:
        s = client.status(f)[0]
        if s.text_status == pysvn.wc_status_kind.modified:
            status = 'M'
        elif s.text_status == pysvn.wc_status_kind.added:
            status = 'M'
        elif s.text_status == pysvn.wc_status_kind.unversioned:
            status = 'U'
        elif s.text_status == pysvn.wc_status_kind.normal:
            status = 'N'
        else:
            print s.text_status
            status = '?'
    except:
        status = '-'
    if status != 'N':
        check[status].append(f)
    
def printfiles(files):
    print '  '+'\n  '.join(files)

def check_files(files):
    [ get_status(f) for f in files ]

    print "\nFiles in unversioned paths:"
    printfiles(check['-'])
    print "\nUnversioned files:"
    printfiles(check['U'])
    print "\nModified or added files:"
    printfiles(check['M'])
    print "\nFiles with unknown status:"
    printfiles(check['?'])
    

if __name__ == '__main__':

    files = get_manifest_files()
    print "Checking %s manifest files" % len(files)
    check_files(files)


# End

