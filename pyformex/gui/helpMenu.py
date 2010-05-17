# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Display help"""

import pyformex as GD

import os,sys
import draw
import utils
import tempfile
import random
import viewport
from gettext import gettext as _

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
    pid = utils.spawn(' '.join([browser,page]))


def catchAndDisplay(cmd):
    """Catch stdout from a Python cmd and display it in a window."""
    save = sys.stdout
    try:
        f = tempfile.TemporaryFile('w+')
        sys.stdout = f
        eval(cmd)
        f.seek(0)
        draw.showText(f.read())
    finally:
        sys.stdout = save
        

def cmdline():
    """Display the pyFormex command line help."""
    catchAndDisplay('GD.print_help()')

## def qappargs():
##     """Display informeation on the Qt application arguments."""
##     qtversion = utils.hasModule('pyqt4')
##     qtversion = '.'.join(qtversion.split('.')[:2])
##     link = "http://doc.trolltech.com/%s/qapplication.html#QApplication" % qtversion
##     help(link)

def opengl():
    """Display the OpenGL format description."""
    draw.showText(viewport.OpenGLFormat())

def detected():
    """Display the detected software components."""
    utils.checkModule()
    utils.checkExternal()
    draw.showText(utils.reportDetected())

def about():
    """Display short information about pyFormex."""
    draw.showInfo("""..

%s
%s

A tool for generating, manipulating and transforming 3D geometrical models by sequences of mathematical operations.

`Copyright 2004-2009 Benedict Verhegghe`

Distributed under the GNU GPL version 3 or later
""" % (GD.Version,'='*len(GD.Version)))

_developers = [
    'Matthieu De Beule',
    'Gianluca De Santis',
    'Bart Desloovere',
    'Peter Mortier',
    'Tim Neels',
    'Tomas Praet',
    'Sofie Van Cauter',
    'Benedict Verhegghe',
    ]

def developers():
    """Display the list of developers."""
    random.shuffle(_developers)
    draw.showInfo("""
The following people have
contributed to pyFormex.
They are listed in random order.

%s

If you feel that your name was left
out in error, please write to
benedict.verhegghe@ugent.be.
""" % '\n'.join(_developers))

                  
_cookies = [
    "Smoking may be hazardous to your health.",
    "Windows is a virus.",
    "Coincidence does not exist. Perfection does.",
    "It's all in the code.",
    "Python is the universal glue.",
    "Intellectual Property Is A Mental Illness.",
    "Programmers are tools for converting caffeine into code.",
    "There are 10 types of people in the world: those who understand binary, and those who don't.",
    "Linux: the choice of a GNU generation",
    "Everything should be made as simple as possible, but not simpler (A. Einstein)",
    
    ]    
random.shuffle(_cookies)

def roll(l):
    l.append(l.pop(0))

def cookie():
    draw.showInfo(_cookies[0],["OK"])
    roll(_cookies)

def links(link):
    help('http://'+link)


def createMenuData():
    """Returns the help menu data"""
    DocsMenuData = [(k,help,{'data':v}) for k,v in GD.cfg['help/docs']] 
    Docs2MenuData = [(k,draw.showFile,{'data':v}) for k,v in GD.cfg['help/docs2']]
    LinksMenuData = [(k,links,{'data':v}) for k,v in GD.cfg['help/links']]

    try:
        MenuData = DocsMenuData + [
            ('---',None),
            (_('&Command line options'),cmdline),
            ] + Docs2MenuData + [
            (_('&Detected Software'),detected), 
            (_('&OpenGL Format'),opengl), 
            (_('&Fortune Cookie'),cookie),
            (_('&Favourite Links'),LinksMenuData),
            (_('&Developers'),developers), 
            (_('&About'),about), 
            ]
    except:
        MenuData = []

    return MenuData


# End
