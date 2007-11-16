#!/usr/bin/env pyformex --gui
# $Id$

from formex import *  # Needed if we want to use this example as a module
from gui.draw import *  # Needed if we want to use this example as a module
import simple
from datetime import datetime
from PyQt4 import QtCore

class AnalogClock(object):
    """An analog clock built from Formices"""

    def __init__(self,lw=2,mm=0.75,hm=0.85,mh=0.7,hh=0.6, sh=0.9):
        """Create an analog clock."""
        self.linewidth = lw
        self.circle = simple.circle(a1=2.,a2=2.)
        radius = Formex(pattern('2'))
        self.mainmark = radius.divide([mm,1.0])
        self.hourmark = radius.divide([hm,1.0])
        self.mainhand = radius.divide([0.0,mh])
        self.hourhand = radius.divide([0.0,hh])
        if sh > 0.0:
            self.secshand = radius.divide([0.0,sh])
        else:
            self.secshand = None
        self.hands = []
        self.timer = None

        
    def draw(self):
        """Draw the clock (without hands)"""
        draw(self.circle,color='black',linewidth=self.linewidth)
        draw(self.mainmark.rosette(4,90),color='black',linewidth=self.linewidth)
        draw(self.hourmark.rot(30).rosette(2,30).rosette(4,90),
             color='black',linewidth=0.5*self.linewidth)


    def drawTime(self,hrs,min,sec=None):
        """Draw the clock's hands showing the specified time.

        If no seconds are specified, no seconds hand is drawn.
        """
        hrot = - hrs*30. - min*0.5
        mrot = - min*6.
        GD.canvas.removeActors(self.hands)
        MH = draw(self.mainhand.rot(mrot),bbox=None,color='red',linewidth=self.linewidth)
        HH = draw(self.hourhand.rot(hrot),bbox=None,color='red',linewidth=self.linewidth)
        self.hands = [MH,HH]
        if self.secshand and sec:
            srot = - sec*6.
            SH = draw(self.secshand.rot(srot),bbox=None,color='orange',linewidth=0.5*self.linewidth)
            self.hands.append(SH)

            
    def drawNow(self):
        """Draw the hands showing the current time."""
        now = datetime.now()
        self.drawTime(now.hour,now.minute,now.second)


    def run(self,granularity=1,runtime=100):
        """Run the clock for runtime seconds, updating every granularity."""
        if granularity > 0.0:
            self.timer = QtCore.QTimer()
            self.timer.connect(self.timer,QtCore.SIGNAL("timeout()"),self.drawNow)
            self.timer.start(1000*granularity)
        if runtime > 0.0:
            self.timeout = QtCore.QTimer()
            self.timeout.connect(self.timeout,QtCore.SIGNAL("timeout()"),self.stop)
            self.timeout.setSingleShot(True)
            self.timeout.start(1000*runtime)

        
    def stop(self):
        """Stop a running clock."""
        if self.timer:
            self.timer.stop()

            
if __name__ == "draw":
    C = AnalogClock()
    C.draw()
    zoomAll()
    C.drawNow()

    if ack("Shall I start the clock?"):
        C.run()
        warning("Please wait until the clock stops running")


