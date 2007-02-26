#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
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

def description():
    """Display the pyFormex description."""
    draw.about(file(GD.cfg['help/description']).read())

def about():
    draw.about("""%s

A tool for generating large 3D structures by mathematical transfomations.

Copyright 2004 Benedict Verhegghe.
Distributed under the GNU GPL.
""" % GD.Version)

def testwarning():
    draw.info("Smoking may be hazardous to your health!\nWindows is a virus!",["OK"])
