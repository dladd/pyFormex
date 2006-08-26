#!/usr/bin/env python
# $Id$
"""Display help"""

import globaldata as GD

import os
import draw

##import helpviewer
##import qt
    
def help(page=None):
    """Display a html help page.

    If GD.help.viewer == None, the help page is displayed using the
    built-in help browser. GD.help.viewer can be set to a string to
    display the page in an external browser.
    """
    if not page:
        page = GD.cfg.help.homepage
    if GD.cfg.help.viewer:
        print 
        pid = os.spawnlp(os.P_NOWAIT,GD.cfg.help.viewer,
                         os.path.basename(GD.cfg.help.viewer),page)
    else:
        if GD.help == None or GD.help.destroyed:
            GD.help = helpviewer.HelpViewer(home = page,
                                            path = os.path.dirname(page),
                                            histfile = GD.cfg.help.history,
                                            bookfile = GD.cfg.help.bookmarks)
            GD.help.setCaption("pyFormex - Helpviewer")
            GD.help.setAbout("pyFormex Help",
                          "This is the pyFormex HelpViewer.<p>It was modeled after the HelpViewer example from the Qt documentation.</p>")
            #help.resize(800,600)
        GD.help.show()


def about():
    draw.about(GD.Version+"""
pyFormex is a python implementation of Formex algebra

http://pyformex.berlios.de

Copyright 2004 Benedict Verhegghe
Distributed under the GNU General Public License.

For help or information, mailto benedict.verhegghe@ugent.be
""")


def testwarning():
    draw.info("Smoking may be hazardous to your health!\nWindows is a virus!",["OK"])
