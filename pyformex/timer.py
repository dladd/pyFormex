#!/usr/bin/env python
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

        This returns the elapsed time since the last reset (or the creation
        of the timer) as a datetime.timedelta object.
        """
        now = datetime.now()
        return now - self.start
    
    def seconds(self,rounded=True):
        """Return the timer readings in seconds.

        The default return value is a rounded integer number of seconds.
        With ``rounded == False``, a floating point value with granularity of
        1 microsecond is returned.
        """
        e = self.read()
        tim = e.days*24*3600 + e.seconds + e.microseconds / 1000000.
        if rounded:
            tim = int(round(tim))
        return tim
        

if __name__ == "__main__":

    import time
    
    t = Timer()
    time.sleep(14.2)
    r = t.read()
    print(r.days,r.seconds,r.microseconds)
    print(t.seconds())
    print(t.seconds(False))

# End
