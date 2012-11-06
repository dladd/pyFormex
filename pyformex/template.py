# pyformex script/app template
#
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

A pyFormex script is just any simple Python source code file with
extension '.py' and is fully read and execution at once.

A pyFormex app can be a '.py' of '.pyc' file, and should define a function
'run()' to be executed by pyFormex. Also, the app should import anything that
it needs.

This template is a common structure that allows the file to be used both as
a script or as an app, with almost identical behavior.

For more details, see the user guide under the `Scripting` section.

The script starts by preference with a docstring (like this),
composed of a short first line, then a blank line and
one or more lines explaining the intention of the script.
"""
from __future__ import print_function

from gui.draw import *  # for an app we need to import explicitely

def run():
    """Main function.

    This is executed on each run.
    """
    print("This is the pyFormex template script/app")


# Initialization code

print("This is the initialization code of the pyFormex template script/app")


# The following is to make script and app behavior alike
if __name__ == 'draw':
    print("Running as a script")
    run()


# End
