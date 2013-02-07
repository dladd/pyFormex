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

def gitRepo(path='.'):
    import git
    return git.Repo(path)


def gitFileStatus(repo,files=None):
    status = {}
    st = repo.git.status(porcelain=True)
    #print st
    for line in st.split('\n'):
        st = line[:2]
        fn = line[3:]
        if files is not None and fn not in files:
            st = 'ND'
        if st not in status:
            status[st] = []
        status[st].append(fn)
    return status

def get_manifest_files():
    return [ f.strip('\n') for f in open('MANIFEST').readlines() if not f.startswith('#')]




def printfiles(files):
    print '  '+'\n  '.join(files)


def filterFileStatus(status,files):
    for st in status:
        status[st] = [ f for f in status[st] if f in files ]
    return status


def checkFileStatus(status):
    check = {
        ' M': 'Modified files',
        '??': 'Untracked files',
        'ND': 'Undistributed files',
        'NF': 'Unfound files',
        }

    for st in check:
        if st in status:
            print '\n'+check[st]+':'
            printfiles(status[st])


if __name__ == '__main__':

    files = get_manifest_files()
    print "Checking %s manifest files" % len(files)

    repo = gitRepo()
    status = gitFileStatus(repo)
    #print status
    checkFileStatus(status)


# End

