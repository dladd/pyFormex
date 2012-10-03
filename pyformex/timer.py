#!/usr/bin/python
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
"""A timer class."""

from datetime import datetime

class Timer(object):
    """A class for measuring elapsed time.
from __future__ import print_function

    A Timer object measures elapsed real time since a specified time, which
    by default is the time of the creation of the Timer.

    Parameters:

    - `start`: a datetime object. If not specified, the time of the creation
      of the Timer is used.    
    """

    def __init__(self,start=None):
        """Create and start a timer."""
        self.reset(start)

    def reset(self,start=None):
        """(Re)Start the timer.

        Sets the start time of the timer to the specified value, or to
        the current time by default.

        Parameters:

        - `start`: a datetime object. If not specified, the current time as
          returned by datetime.now() is used.
        """
        if isinstance(start,datetime):
            self.start = start
        else:
            self.start = datetime.now()

    def read(self,reset=False):
        """Read the timer.

        Returns the elapsed time since the last reset (or the creation
        of the timer) as a datetime.timedelta object.

        If reset=True, the timer is reset to the time of reading.
        """
        now = datetime.now()
        ret = now - self.start
        if reset:
            self.start = now
        return ret
    
    def seconds(self,reset=False,rounded=True):
        """Return the timer readings in seconds.

        The default return value is a rounded integer number of seconds.
        With ``rounded == False``, a floating point value with granularity of
        1 microsecond is returned.

        If reset=True, the timer is reset at the time of reading.
        """
        e = self.read(reset)
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
