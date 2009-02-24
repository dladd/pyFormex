## $Id$
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
"""A locking mechanism for the4 drawing functions."""

import pyformex as GD

import threading


class DrawLock(object):
    """A timed lock to slow down drawing processes"""

    def __init__(self):
        self.allowed = True
        self.locked = False
        self.timer = None


    def wait(self):
        """Wait for the drawing lock to be released.
        
        This method can be called to wait until the lock is released,
        while still processing GUI events.
        """
        if self.allowed:
            while self.locked:
                GD.app.processEvents()
                GD.canvas.update()


    def lock(self,time=None):
        """Lock the drawing function for the next time seconds.

        If a no time is specified, a global value is used.
        """
        #GD.debug("LOCKING: %s" % time)
        if self.allowed and not self.locked:
            if time is None:
                time = GD.GUI.drawwait
            if time > 0:
                self.locked = True
                self.timer = threading.Timer(time,self.release)
                self.timer.start()


    def block(self):
        """Lock the drawing function indefinitely."""
        if self.timer:
            self.timer.cancel()
        self.locked = True


    def release(self):
        """Release the lock on the drawing function.

        If a timer is running, cancel it.
        """
        self.locked = False
        if self.timer:
            self.timer.cancel()


    def free(self):
        """Release the lock and prevent waits until allow() is called."""
        self.allowed = False
        self.release()


    def allow(self):
        """Allow draw waits.

        This is called after a free() to reinstall the draw locking.
        """
        self.allowed = True

#### End
