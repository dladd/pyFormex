#!/usr/bin/env python
# $Id$
"""Display help"""

import globaldata as GD

import os
import draw
import utils

def help(page=None):
    """Display a html help page.

    If GD.help.viewer == None, the help page is displayed using the
    built-in help browser. GD.help.viewer can be set to a string to
    display the page in an external browser.
    """
    if not page:
        page = GD.cfg['help/manual']
    if page.startswith('http:'):
        browser = GD.cfg['browser']
    else:
        browser = GD.cfg['viewer']
    pid = utils.spawn(browser % page)


def manual():
    """Display the pyFormex manual."""
    help(GD.cfg['help/manual'])

def pydoc():
    """Display the pydoc information."""
    help(GD.cfg['help/pydocs'])

def website():
    """Display the pyFormex website."""
    help(GD.cfg['help/website'])

def about():
    s = GD.Version
    desc = GD.cfg['help/about']
    if os.path.exists(desc):
        s += file(desc,'r').readlines()
    draw.about(s)

def testwarning():
    draw.info("Smoking may be hazardous to your health!\nWindows is a virus!",["OK"])
