## $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
"""A locking mechanism for the drawing functions.

"""
from __future__ import print_function
import pyformex as pf

import threading
import time


# WE SHOULD REIMPLEMENT THIS USING PYTHON threading.Condition?

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
                pf.canvas.update()
                pf.app.processEvents()
                time.sleep(0.01)  # to avoid overusing the cpu


    def lock(self,time=None):
        """Lock the drawing function for the next time seconds.

        If a no time is specified, a global value is used.
        """
        #print self.allowed,self.locked
        if self.allowed and not self.locked:
            if time is None:
                time = pf.GUI.drawwait
            if time > 0.:
                pf.debug('STARTING TIMER',pf.DEBUG.SCRIPT)
                self.locked = True
                self.timer = threading.Timer(time,self.release)
                self.timer.start()


    # ?? SHOULD block and release only be activated if self.allowed ??
    def block(self):
        """Lock the drawing function indefinitely."""
        if self.timer:
            self.timer.cancel()
        self.locked = True


    def release(self):
        """Release the lock on the drawing function.

        If a timer is running, cancel it.
        """
        #print "RELEASING LOCK"
        self.locked = False
        if self.timer:
            self.timer.cancel()


    def free(self):
        """Release the lock and prevent waits until allow() is called."""
        #print "FREEING THE DRAW LOCK"
        self.allowed = False
        self.release()


    def allow(self):
        """Allow draw waits.

        This is called after a free() to reinstall the draw locking.
        """
        self.allowed = True


class Repeater(object):
    """Repeatedly execute a function.

    The Repeater class provides functionality to repeatedly execute a
    function, while allowing the GUI to process events so that user
    interactivity can continue. It also avoids using too much CPU time
    while running empty. The function can be repeated until one of the
    following conditions is met:

    - the called function returns a value that evaluates to True
    - a specified time has elapsed
    - a number of executins has been reached
    - and external event stops the execution

    Parameters:

    - `func`: if callable, this function will be called repeatedly why
      the Repeater class is active. The function will be passed all
      the extra parameters *args and **kargs. If the function returns
      a value that does not evaluate to False, execution is halted.
    - `duration`: max duration for the repeated execution. If < 0, repeats
      indefinitely.
    - `maxcount`: max number of executions. If < 0, there is no limit.
    - `sleep`: extra time to sleep between two executions. The actual time
      between two executions may be higher, because any GUI events will
      also be executed. If your function does not do anything, setting a
      value > 0 is recommended to avoid high CPU usage while running idle.

    Execution is started by calling the start() method. The method returns
    after some event has made it to stop, with an exitcode of:

    - 1, if the stop() method was used
    - 2, if a timeout occurred
    - 3, if the maximum number of excutions occurred,
    - or else, with the value returned by the function.

    """
    def __init__(self,func,duration=-1,maxcount=-1,sleep=0,*args,**kargs):
        """Create a new repeater"""
        pf.debug("REPEAT: %s, %s" % (duration,maxcount),pf.DEBUG.SCRIPT)
        self.exitcode = False
        self.func = func
        self.duration = duration
        self.maxcount = maxcount
        self.sleep = sleep

    ## def start(self):
    ##     """Start repeated execution"""
        timer = None
        if self.duration >= 0:
            timer = threading.Timer(self.duration,self.timeOut)
            timer.start()
        self.exitcode = 0
        count = 0
        while not self.exitcode:
            pf.app.processEvents()
            if callable(self.func):
                self.exitcode = self.func(*args,**kargs)
                break
            count += 1
            if self.maxcount >= 0 and count >= self.maxcount:
                self.exitcode = 3
                break
            if self.sleep > 0:
                time.sleep(self.sleep)

        pf.debug("Exit Repeater with Exitcode %s, Count: %s" % (self.exitcode,count),pf.DEBUG.SCRIPT)

    def timeOut(self):
        """Stop the repeater because of a timeout"""
        self.exitcode = 2

    def stop(self,exitcode=1):
        """Interrupt a repeater with given exitcode"""
        self.exitcode = exitcode



#### End
