## $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
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
        GD.debug("LOCKING: %s" % time)
        if self.allowed and not self.locked:
            if time is None:
                time = GD.gui.drawwait
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
