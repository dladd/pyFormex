# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Functions for the Pref menu."""

import globaldata as GD
import os

from gettext import gettext as _
import utils
import widgets
import draw
import imageViewer


def askConfigPreferences(items,prefix=None,store=None):
    """Ask preferences stored in config variables.

    Items in list should only be keys. store is usually a dictionary, but
    can be any class that allow the setdefault method for lookup while
    setting the default, and the store[key]=val syntax for setting the
    value.
    If a prefix is given, actual keys will be 'prefix/key'. 
    The current values are retrieved from the store, and the type returned
    will be in accordance.
    If no store is specified, the global config GD.cfg is used.
    """
    if not store:
        store = GD.cfg
    if prefix:
        items = [ '%s/%s' % (prefix,i) for i in items ]
    itemlist = [ [ i,store.setdefault(i,'') ] for i in items ]
    res,accept = widgets.InputDialog(itemlist,'Config Dialog',GD.gui).getResult()
    if accept:
        GD.debug(res)
        if draw.ack("Update the settings?"):
            # This does not work for our Config class!
            # store.update(res)
            # Therefore, set individually
            for k,v in res.items():
                store[k] = v
##        for i,r in zip(itemlist,res):
##            GD.debug("IN : %s\nOUT: %s" % (i,r))
##            if type(i[1]) == str:
##                store[r[0]] = r[1]
##            else:
##                store[r[0]] = eval(r[1])
        GD.debug(GD.cfg)
    return accept


def setToolbarPlacement(store=None):
    """Ask placement of toolbars.

    Items in list should be existing toolbar widgets.
    """
    if store is None:
        store = GD.cfg
    toolbar = [ GD.gui.modebar, GD.gui.viewbar ]
    setting = ['gui/modebar', 'gui/viewbar' ]
    options = [ None, 'default', 'left', 'right', 'top', 'bottom' ]
    label = [ str(tb.windowTitle()) for tb in toolbar ]
    current = [ store[s] for s in setting ]
    itemlist = [(l, options[1], 'select', options) for (l,c) in zip(label,setting)]
    itemlist.append(('Store these settings as defaults', False))
    res,accept = widgets.InputDialog(itemlist,'Config Dialog',GD.gui).getResult()
    if accept:
        GD.debug(res)
        if res['Store these settings as defaults']:
            # The following  does not work for our Config class!
            #    store.update(res)
            # Therefore, we set the items individually
            for s,l in zip(setting,label):
                val = res[l]
                if val == "None":
                    val = None
                store[s] = val
        GD.debug(store)
    return accept


def setHelp():
    askConfigPreferences(['viewer','help/manual','help/pydocs'])

def setCommands():
    askConfigPreferences(['editor','viewer','browser'])

def setSysPath():
    askConfigPreferences(['syspath'])

def setInputTimeout():
    askConfigPreferences(['input/timeout'])

def setDrawWait():
    askConfigPreferences(['draw/wait'])

def setBGcolor():
    col = GD.cfg['draw/bgcolor']
    col = widgets.getColor(col)
    if col:
        GD.cfg['draw/bgcolor'] = col

def setLinewidth():
    askConfigPreferences(['draw/linewidth'])

def setSize():
    GD.gui.resize(800,600)

def setPickSize():
    w,h = GD.cfg['pick/size']
    res = draw.askItems([['w',w],['h',h]])
    GD.cfg['pick/size'] = (int(res['w']),int(res['h']))
        
    
def setRender():
    if askConfigPreferences(['ambient', 'specular', 'emission', 'shininess'],'render'):
        draw.smooth()

def setLight(light=0):
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    tgt = 'render/light%s'%light
    localcopy = {}
    localcopy.update(GD.cfg[tgt])
    if askConfigPreferences(keys,store=localcopy):
        GD.cfg[tgt] = localcopy
        draw.smooth()

def setLight0():
    setLight(0)

def setLight1():
    setLight(1)

def setRotFactor():
    askConfigPreferences(['gui/rotfactor'])
def setPanFactor():
    askConfigPreferences(['gui/panfactor'])
def setZoomFactor():
    askConfigPreferences(['gui/zoomfactor'])


def setFont(font=None):
    """Ask and set the main application font."""
    if font is None:
        font = widgets.selectFont()
    if font:
        GD.cfg['gui/font'] = str(font.toString())
        GD.cfg['gui/fontfamily'] = str(font.family())
        GD.cfg['gui/fontsize'] = font.pointSize()
        GD.gui.setFont(font)


def setAppearance():
    """Ask and set the main application appearence."""
    style,font = widgets.AppearenceDialog().getResult()
    if style:
        # Get style name, strip off the leading 'Q' and trailing 'Style'
        stylename = style.metaObject().className()[1:-5]
        GD.cfg['gui/style'] = stylename
        GD.gui.setStyle(stylename)
    if font:
        setFont(font)


def setSplash():
    """Open an image file and set it as the splash screen."""
    cur = GD.cfg['gui/splash']
    if not cur:
        cur = GD.cfg.get('icondir','.')
    w = widgets.ImageViewerDialog(path=cur)
    fn = w.getFilename()
    w.close()
    if fn:
        GD.cfg['gui/splash'] = fn
      
    

MenuData = [
    (_('&Settings'),[
        (_('&Appearance'),setAppearance), 
        (_('&Font'),setFont), 
        (_('&Toolbar Placement'),setToolbarPlacement), 
        (_('&Input Timeout'),setInputTimeout), 
        (_('&Draw Wait Time'),setDrawWait), 
#        (_('&Background Color'),setBGcolor), 
#        (_('Line&Width'),setLinewidth), 
        (_('&Pick Size'),setPickSize), 
        (_('&RotFactor'),setRotFactor),
        (_('&PanFactor'),setPanFactor),
        (_('&ZoomFactor'),setZoomFactor),
        (_('&Rendering'),setRender),
        (_('&Light0'),setLight0),
        (_('&Light1'),setLight1),
        (_('&Splash Image'),setSplash),
        (_('&Commands'),setCommands),
        (_('&SysPath'),setSysPath),
        (_('&Help'),setHelp),
        ('---',None),
        (_('&Save Preferences'),GD.savePreferences),
        (_('Toggle Timeout'),draw.timeout),
        ]),
    ]


   
# End
