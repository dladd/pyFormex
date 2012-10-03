#        *** pyformex ***
##
##  Copyright (C) 2011 John Doe (j.doe@somewhere.org) 
##  Distributed under the GNU General Public License version 3 or later.
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

"""pyFormex Script/App Template

This is a template file to show the general layout of a pyFormex
script or app.

In the current version, a pyFormex script should obey the following rules:

- file name extension is '.py'
- first (comment) line contains 'pyformex'

A pyFormex app can be a '.py' of '.pyc' file, and should define a function
'run()' to be executed by pyFormex. Also, the app should import anythin that
it needs.

This template is a common structure that allows the file to be used both as
a script or as an app.

The script starts by preference with a docstring (like this),
composed of a short first line, then a blank line and
one or more lines explaining the intention of the script.
"""
from __future__ import print_function
from gui.draw import *

def run():
    print("This is the pyFormex template script/app")

# The following is to make it work as a script
if __name__ == 'draw':
    run()


# End
