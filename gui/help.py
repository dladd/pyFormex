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

    If no page is specified, the help manual is displayed.

    If page is a string starting with 'http:', the page is displayed with
    the command set in GD.cfg['browser'], else with the command in
    GD.cfg['viewer']
    """
    if not page:
        page = GD.cfg['help/manual']
    if page.startswith('http:'):
        browser = GD.cfg['browser']
    else:
        browser = GD.cfg['viewer']
    pid = utils.spawn(browser % page)


def cmdline():
    """Display the pyFormex command line help."""
    GD.print_help()

def manual():
    """Display the pyFormex manual."""
    help(GD.cfg['help/manual'])

def pydoc():
    """Display the pydoc information."""
    help(GD.cfg['help/pydocs'])

def website():
    """Display the pyFormex website."""
    help(GD.cfg['help/website'])

def webman():
    """Display the pyFormex website."""
    help(GD.cfg['help/webmanual'])

def readme():
    """Display the pyFormex description."""
    draw.textView(file(GD.cfg['help/readme']).read())

def license():
    """Display the pyFormex description."""
    draw.textView(file(GD.cfg['help/license']).read())

def about():
    draw.about("""%s

A tool for generating and operating on large 3D structures by mathematical transfomations.

Copyright 2004-2007 Benedict Verhegghe.
Distributed under the GNU GPL v2 or higher.
""" % GD.Version)

def testwarning():
    draw.info("Smoking may be hazardous to your health!\nWindows is a virus!",["OK"])
