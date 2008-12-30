#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""A timer class."""

from datetime import datetime

class Timer(object):
    """A class for measuring elapsed time."""

    def __init__(self):
        """Create and start a timer."""
        self.reset()

    def reset(self):
        """Start the timer."""
        self.start = datetime.now()

    def read(self):
        """Read the timer.

        This returns the elapsed time since the last reset or then creation
        of the timer as a datetime.timedelta object.
        """
        now = datetime.now()
        return now - self.start
    
    def seconds(self):
        """Return the timer readings in seconds."""
        e = self.read()
        return e.days*24*3600 + e.seconds + int(round(e.microseconds / 1000000.))
        

if __name__ == "__main__":

    import time
    
    t = Timer()
    time.sleep(5)
    r = t.read()
    print r.days,r.seconds,r.microseconds
    print t.seconds()

# End
