#!/usr/bin/env python
# $Id $

# This file is intended to disappear, after its contents has been
# moved to a more appropriate place.

import globaldata as GD
import gui
import draw
import widgets
import canvas
import help

import sys,time,os,string
import qt
import qtgl


def NotImplemented():
    draw.warning("This option has not been implemented yet!")

#####################################################################
# Opening, Playing and Saving pyformex scripts

save = NotImplemented
saveAs = NotImplemented

def editor():
    if GD.gui.editor:
        print "Close editor"
        GD.gui.closeEditor()
    else:
        print "Open editor"
        GD.gui.showEditor()

############################################################################

# JUST TESTING:
def userView(i=1):
    if i==1:
        frontView()
    else:
        isoView()

     


#### End
