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
"""Functions for the Pref menu."""

import pyformex
from pyformex.main import savePreferences
import os

from gettext import gettext as _
import utils
import widgets
import toolbar
import draw
import imageViewer

        

def updateSettings(res,save=None):
    """Update the current settings (store) with the values in res.

    res is a dictionary with configuration values.
    The current settings will be update with the values in res.

    If res contains a key 'Save changes' and its value is True, the
    preferences will also be saved to the user's preference file.
    Else, the user will be asked whether he wants to save the changes.
    """
    pyformex.debug("Accepted settings:",res)
    if save is None:
        save = res.get('Save changes',None)
    if save is None:
        save = ack("Save the current changes to your configuration file?")

    # Do not use 'pyformex.cfg.update(res)' here!
    # It will not work with our Config class!

    for k in res:
        if k.startswith('_'): # skip temporary variables
            continue
        
        pyformex.cfg[k] = res[k]
        if save and pyformex.prefcfg[k] != pyformex.cfg[k]:
            pyformex.prefcfg[k] = pyformex.cfg[k]

        if pyformex.GUI:
            if k in _activate_settings:
                _activate_settings[k]()

    pyformex.debug("New settings:",pyformex.cfg)
    pyformex.debug("New preferences:",pyformex.prefcfg)


def settings():
    import plugins
    
    dia = None

    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = dia.results
        res['Save changes'] = save
        #print res
        ok_plugins = utils.subDict(res,'_plugins/')
        #print ok_plugins
        res['gui/plugins'] = [ p for p in ok_plugins if ok_plugins[p]]
        updateSettings(res)
        pyformex.cfg.update(res)
        plugins.loadConfiguredPlugins()

    def acceptAndSave():
        accept(save=True)

    def autoSettings(keylist):
        return [(k,pyformex.cfg[k]) for k in keylist]

    mouse_settings = autoSettings(['gui/rotfactor','gui/panfactor','gui/zoomfactor','gui/autozoomfactor','gui/dynazoom','gui/wheelzoom'])

    plugin_items = [ ('_plugins/'+name,name in pyformex.cfg['gui/plugins'],{'text':label}) for (label,name) in plugins.plugin_menus ]

    dia = widgets.InputDialog(
        caption='pyFormex Settings',items={
            'General':[
                ('syspath',pyformex.cfg['syspath']),
                ('editor',pyformex.cfg['editor']),
                ('viewer',pyformex.cfg['viewer']),
                ('browser',pyformex.cfg['browser']),
                ('help/docs',pyformex.cfg['help/docs']),
                ('autorun',pyformex.cfg['autorun'],{'text':'Startup script','tooltip':'This script will automatically be run at pyFormex startup'}),
                ],
            'Mouse': mouse_settings,
            'GUI':[
                ('gui/coordsbox',pyformex.cfg['gui/coordsbox']),
                ('gui/showfocus',pyformex.cfg['gui/showfocus']),
                ('gui/timeoutbutton',pyformex.cfg['gui/timeoutbutton']),
                ('gui/timeoutvalue',pyformex.cfg['gui/timeoutvalue']),
                ],
            'Plugins': plugin_items,
            },
        actions=[
            ('Close',close),
            ('Accept and Save',acceptAndSave),
            ('Accept',accept),
        ])
    dia.show()


def askConfigPreferences(items,prefix=None,store=None):
    """Ask preferences stored in config variables.

    Items in list should only be keys. store is usually a dictionary, but
    can be any class that allow the setdefault method for lookup while
    setting the default, and the store[key]=val syntax for setting the
    value.
    If a prefix is given, actual keys will be 'prefix/key'. 
    The current values are retrieved from the store, and the type returned
    will be in accordance.
    If no store is specified, the global config pyformex.cfg is used.
    """
    if store is None:
        store = pyformex.cfg
    if prefix:
        items = [ '%s/%s' % (prefix,i) for i in items ]
    itemlist = [ [ i,store.setdefault(i,'') ] for i in items ]
    res = widgets.InputDialog(itemlist+[('Save changes',True)],'Config Dialog',pyformex.GUI).getResult()
    if res and store==pyformex.cfg:
        updateSettings(res)
    return res


def setToolbarPlacement(store=None):
    """Ask placement of toolbars.

    Items in list should be existing toolbar widgets.
    """
    if store is None:
        store = pyformex.cfg
    toolbar = pyformex.GUI.toolbardefs
    label = [ i[0] for i in toolbar ]
    setting = [ 'gui/%s' % i[1] for i in toolbar ]
    options = [ None, 'default', 'left', 'right', 'top', 'bottom' ]
    current = [ store[s] for s in setting ]
    itemlist = [(l, options[1], 'select', options) for (l,c) in zip(label,setting)]
    itemlist.append(('Store these settings as defaults', False))
    res = widgets.InputDialog(itemlist,'Config Dialog',pyformex.GUI).getResult()
    if res:
        pyformex.debug(res)
        if res['Store these settings as defaults']:
            # The following  does not work for our Config class!
            #    store.update(res)
            # Therefore, we set the items individually
            for s,l in zip(setting,label):
                val = res[l]
                if val == "None":
                    val = None
                store[s] = val
        pyformex.debug(store)

 
def setDrawWait():
    askConfigPreferences(['draw/wait'])
    pyformex.GUI.drawwait = pyformex.cfg['draw/wait']

def setLinewidth():
    askConfigPreferences(['draw/linewidth'])


def setAvgNormalTreshold():
    askConfigPreferences(['render/avgnormaltreshold'])
def setAvgNormalSize():
    askConfigPreferences(['mark/avgnormalsize'])

def setSize():
    pyformex.GUI.resize(800,600)

def setPickSize():
    w,h = pyformex.cfg['pick/size']
    res = draw.askItems([['w',w],['h',h]])
    pyformex.cfg['pick/size'] = (int(res['w']),int(res['h']))
        
    
def setRenderMode():
    from canvas import Canvas
    res = draw.askItems([('render/mode',None,'vradio',{'text':'Render Mode','choices':Canvas.rendermodes})])
    if res:
        rendermode = res['render/mode']
        if hasattr(draw,rendermode):
            getattr(draw,rendermode)()
        updateSettings(res)
            
    
def setRender():
    items = [ ('render/%s'%a,getattr(pyformex.canvas,a),'slider',{'min':0,'max':100,'scale':0.01,'func':getattr(draw,'set_%s'%a)}) for a in [ 'ambient', 'specular', 'emission', 'shininess' ] ]
    res = draw.askItems(items)
    if res:
        updateSettings(res)


def setLight(light=0):
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    tgt = 'render/light%s'%light
    localcopy = {}
    localcopy.update(pyformex.cfg[tgt])
    if askConfigPreferences(keys,store=localcopy):
        pyformex.cfg[tgt] = localcopy
        draw.smooth()

def setLight0():
    setLight(0)

def setLight1():
    setLight(1)
        


def setFont(font=None):
    """Ask and set the main application font."""
    if font is None:
        font = widgets.selectFont()
    if font:
        pyformex.cfg['gui/font'] = str(font.toString())
        pyformex.cfg['gui/fontfamily'] = str(font.family())
        pyformex.cfg['gui/fontsize'] = font.pointSize()
        pyformex.GUI.setFont(font)


def setAppearance():
    """Ask and set the main application appearence."""
    style,font = widgets.AppearenceDialog().getResult()
    if style:
        # Get style name, strip off the leading 'Q' and trailing 'Style'
        stylename = style.metaObject().className()[1:-5]
        pyformex.cfg['gui/style'] = stylename
        pyformex.GUI.setStyle(stylename)
    if font:
        setFont(font)


def setSplash():
    """Open an image file and set it as the splash screen."""
    cur = pyformex.cfg['gui/splash']
    if not cur:
        cur = pyformex.cfg.get('icondir','.')
    w = widgets.ImageViewerDialog(path=cur)
    fn = w.getFilename()
    w.close()
    if fn:
        pyformex.cfg['gui/splash'] = fn

w=None

def setScriptDirs():
    global w
    from scriptMenu import reloadScriptMenu
    scr = pyformex.cfg['scriptdirs']
    w = widgets.Dialog([widgets.Table(scr,chead=['Label','Path'])],
                       title='Script paths',
                       actions=[('New',insertRow),('Edit',editRow),('Delete',removeRow),('Move Up',moveUp),('Reload',reloadScriptMenu),('OK',)])
    w.show()

def insertRow():
    ww = widgets.FileSelection(pyformex.cfg['workdir'],'*',exist=True,dir=True)
    fn = ww.getFilename()
    if fn:
        scr = pyformex.cfg['scriptdirs']
        w.table.model().insertRows()
        scr[-1] = ['New',fn]
    w.table.update()
    
def editRow():
    row = w.table.currentIndex().row()
    scr = pyformex.cfg['scriptdirs']
    item = scr[row]
    res = draw.askItems([('Label',item[0]),('Path',item[1])])
    if res:
        scr[row] = [res['Label'],res['Path']]
    w.table.update()

def removeRow():
    row = w.table.currentIndex().row()
    w.table.model().removeRows(row,1)
    w.table.update()

def moveUp():
    row = w.table.currentIndex().row()
    scr = pyformex.cfg['scriptdirs']
    if row > 0:
        a,b = scr[row-1:row+1]
        scr[row-1] = b
        scr[row] = a
    w.table.setFocus() # For some unkown reason, this seems needed to
                       # immediately update the widget
    w.table.update()
    

## def editConfig():
##     error('You can not edit the config file while pyFormex is running!') 
        

def setOptions():
    options = ['test','debug','uselib','safelib','fastencode']
    options = [ o for o in options if hasattr(pyformex.options,o) ]
    items = [ (o,getattr(pyformex.options,o)) for o in options ]
    res = draw.askItems(items)
    if res:
        for o in options:
            setattr(pyformex.options,o,res[o])
            print(pyformex.options)
            ## if o == 'debug':
            ##     pyformex.setDebugFunc()
    


# Functions defined to delay binding
def coordsbox():
    """Toggle the coordinate display box onor off"""
    pyformex.GUI.coordsbox.setVisible(pyformex.cfg['gui/coordsbox'])
    
def timeoutbutton():
    """Toggle the timeout button on or off"""
    toolbar.addTimeoutButton(pyformex.GUI.toolbar)

def updateCanvas():
    pyformex.canvas.update()

    
# This sets the functions that should be called when a setting has changed
_activate_settings = {
    'gui/coordsbox':coordsbox,
    'gui/timeoutbutton':timeoutbutton,
    'gui/showfocus':updateCanvas,
    }
   

MenuData = [
    (_('&Settings'),[
        (_('&Settings Dialog'),settings), 
        (_('&Options'),setOptions),
        ('---',None),
        (_('&Appearance'),setAppearance), 
        (_('&Font'),setFont), 
        (_('&Toolbar Placement'),setToolbarPlacement), 
        (_('&Draw Wait Time'),setDrawWait), 
        (_('Avg&Normal Treshold'),setAvgNormalTreshold), 
        (_('Avg&Normal Size'),setAvgNormalSize), 
        (_('&Pick Size'),setPickSize), 
        (_('&Render Mode'),setRenderMode),
        (_('&Rendering'),setRender),
        (_('&Light0'),setLight0),
        (_('&Light1'),setLight1),
        (_('&Splash Image'),setSplash),
        (_('&Script Paths'),setScriptDirs),
        ('---',None),
##         (_('&Edit Preferences'),editPreferences),
        (_('&Save Preferences Now'),savePreferences),
#        (_('&Make current settings the defaults'),savePreferences),
        ]),
    ]


   
# End
