#!/usr/bin/env python
# $Id$
"""Display help"""

import globaldata as GD
import helpviewer
import gui
import qt

    
def help():
    """Display the help browser"""
    print "help = ",GD.cfg.help
    if GD.help == None or GD.help.destroyed:
        GD.help = helpviewer.HelpViewer(home = GD.cfg.help.homepage,
                                        path = GD.cfg.help.helpdir,
                                        histfile = GD.cfg.help.history,
                                        bookfile = GD.cfg.help.bookmarks)
        GD.help.setCaption("pyFormex - Helpviewer")
        GD.help.setAbout("pyFormex Help",
                      "This is the pyFormex HelpViewer.<p>It was modeled after the HelpViewer example from the Qt documentation.</p>")
        #help.resize(800,600)
        GD.help.connect(GD.help,qt.SIGNAL("destroyed()"),closeHelp)
    GD.help.show()


def closeHelp():
    """Close the help browser"""
    print "Closing the help window!"
    GD.help = None
        

def about():
    gui.about(GD.Version+"""
pyFormex is a python implementation of Formex algebra

http://pyformex.berlios.de

Copyright 2004 Benedict Verhegghe
Distributed under the GNU General Public License.

For help or information, mailto benedict.verhegghe@ugent.be
""")

def testwarning():
    gui.info("Smoking may be hazardous to your health!\nWindows is a virus!",["OK","Cancel"])
