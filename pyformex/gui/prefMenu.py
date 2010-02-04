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

import pyformex as GD
import os

from gettext import gettext as _
import utils
import widgets
import draw
import imageViewer


def nextOnly():
    draw.warning("Changes to this setting will only become effective after restarting the program.")

def updateSettings(res,store):
    """Update the current settings (store) with the values in res.

    store and res are both dictionaries
    This asks the users to confirm that he wants to update the settings.
    """
    GD.debug(res)
    if res.get('Save changes',None) or draw.ack("Save the changes to your configuration file?\n\nCurrently, the changes will be rejected if you do not save them!\n"):
        # Do not use 'store.update(res)' here!
        # It will not work with our Config class!
        for k,v in res.items():
            store[k] = v
        GD.debug(store)
        return True
    return False


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
    res = widgets.InputDialog(itemlist+[('Save changes',True)],'Config Dialog',GD.GUI).getResult()
    if res:
        updateSettings(res,store)
    return res


def setToolbarPlacement(store=None):
    """Ask placement of toolbars.

    Items in list should be existing toolbar widgets.
    """
    if store is None:
        store = GD.cfg
    toolbar = GD.GUI.toolbardefs
    label = [ i[0] for i in toolbar ]
    setting = [ 'gui/%s' % i[1] for i in toolbar ]
    options = [ None, 'default', 'left', 'right', 'top', 'bottom' ]
    current = [ store[s] for s in setting ]
    itemlist = [(l, options[1], 'select', options) for (l,c) in zip(label,setting)]
    itemlist.append(('Store these settings as defaults', False))
    res = widgets.InputDialog(itemlist,'Config Dialog',GD.GUI).getResult()
    if res:
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

def setTimeoutButton():
    askConfigPreferences(['gui/timeoutbutton'])
    nextOnly()
    
def setShowFocus():
    askConfigPreferences(['gui/showfocus'])
    

def setHelp():
    askConfigPreferences(['help/docs'])

def setCommands():
    askConfigPreferences(['editor','viewer','browser'])

def setSysPath():
    askConfigPreferences(['syspath'])

def setInputTimeout():
    askConfigPreferences(['input/timeout'])

def setDrawWait():
    askConfigPreferences(['draw/wait'])
    GD.GUI.drawwait = GD.cfg['draw/wait']

def setLinewidth():
    askConfigPreferences(['draw/linewidth'])


def setAvgNormalTreshold():
    askConfigPreferences(['render/avgnormaltreshold'])
def setAvgNormalSize():
    askConfigPreferences(['mark/avgnormalsize'])

def setSize():
    GD.GUI.resize(800,600)

def setPickSize():
    w,h = GD.cfg['pick/size']
    res = draw.askItems([['w',w],['h',h]])
    GD.cfg['pick/size'] = (int(res['w']),int(res['h']))
        
    
def setRenderMode():
    from canvas import Canvas
    res = draw.askItems([('render/mode',None,'vradio',{'text':'Render Mode','choices':Canvas.rendermodes})])
    if res:
        rendermode = res['render/mode']
        if hasattr(draw,rendermode):
            getattr(draw,rendermode)()
        updateSettings(res,GD.cfg)
            
    
def setRender():
    items = [ ('render/%s'%a,getattr(GD.canvas,a),'slider',{'min':0,'max':100,'scale':0.01,'func':getattr(draw,'set_%s'%a)}) for a in [ 'ambient', 'specular', 'emission', 'shininess' ] ]
    res = draw.askItems(items)
    if res:
        updateSettings(res,GD.cfg)


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
def setAutoZoomFactor():
    askConfigPreferences(['gui/autozoomfactor'])
def setZoomActions():
    askConfigPreferences(['gui/dynazoom','gui/wheelzoom'])

def setPlugins():
    import plugins
    configured_plugins = GD.cfg['gui/plugins']
    items = [ (name,name in configured_plugins,{'text':label}) for (label,name) in plugins.plugin_menus ]
    #items.append(('apply',True,{'text':'Apply this to the current session'}))
    res = draw.askItems(items)
    print "Activated plugins: %s" % res.keys()
    if res:
        ok_plugins = [ name for name in res if res[name] ]
        GD.cfg['gui/plugins'] = ok_plugins
        


def setFont(font=None):
    """Ask and set the main application font."""
    if font is None:
        font = widgets.selectFont()
    if font:
        GD.cfg['gui/font'] = str(font.toString())
        GD.cfg['gui/fontfamily'] = str(font.family())
        GD.cfg['gui/fontsize'] = font.pointSize()
        GD.GUI.setFont(font)


def setAppearance():
    """Ask and set the main application appearence."""
    style,font = widgets.AppearenceDialog().getResult()
    if style:
        # Get style name, strip off the leading 'Q' and trailing 'Style'
        stylename = style.metaObject().className()[1:-5]
        GD.cfg['gui/style'] = stylename
        GD.GUI.setStyle(stylename)
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

def setAutoRun():
    askConfigPreferences(['autorun'])

w=None

def setScriptDirs():
    global w
    from scriptMenu import reloadScriptMenu
    scr = GD.cfg['scriptdirs']
    w = widgets.Dialog([widgets.Table(scr,chead=['Label','Path'])],
                       title='Script paths',
                       actions=[('New',insertRow),('Edit',editRow),('Delete',removeRow),('Move Up',moveUp),('Reload',reloadScriptMenu),('OK',)])
    w.show()

def insertRow():
    ww = widgets.FileSelection(GD.cfg['workdir'],'*',exist=True,dir=True)
    fn = ww.getFilename()
    if fn:
        scr = GD.cfg['scriptdirs']
        w.table.model().insertRows()
        scr[-1] = ['New',fn]
    w.table.update()
    
def editRow():
    row = w.table.currentIndex().row()
    scr = GD.cfg['scriptdirs']
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
    scr = GD.cfg['scriptdirs']
    if row > 0:
        a,b = scr[row-1:row+1]
        scr[row-1] = b
        scr[row] = a
    w.table.setFocus() # For some unkown reason, this seems needed to
                       # immediately update the widget
    w.table.update()
    

## def editConfig():
##     error('You can not edit the config file while pyFormex is running!') 
    

def savePreferences():
    """Save the preferences.

    The name of the preferences file was set in GD.preffile.
    If a local preferences file was read, it will be saved there.
    Otherwise, it will be saved as the user preferences, possibly
    creating that file.
    If GD.preffile is None, preferences are not saved.
    """
    f = GD.preffile
    if not f:
        return
    
    del GD.cfg['__ref__']

    # Dangerous to set permanently!
    del GD.cfg['input/timeout']
    
    GD.debug("!!!Saving config:\n%s" % GD.cfg)

    try:
        fil = file(f,'w')
        fil.write("%s" % GD.cfg)
        fil.close()
        res = "Saved"
    except:
        res = "Could not save"
    GD.debug("%s preferences to file %s" % (res,f))


def coordsbox():
    """Toggle the coordinate display box onor off"""
    GD.GUI.toggleCoordsTracker()
    

MenuData = [
    (_('&Settings'),[
        (_('&Appearance'),setAppearance), 
        (_('&Font'),setFont), 
        (_('&Toolbar Placement'),setToolbarPlacement), 
        (_('&Show Timeout Button'),setTimeoutButton),
        (_('&Input Timeout'),setInputTimeout), 
        (_('&Draw Wait Time'),setDrawWait), 
        (_('Avg&Normal Treshold'),setAvgNormalTreshold), 
        (_('Avg&Normal Size'),setAvgNormalSize), 
        (_('&Pick Size'),setPickSize), 
        (_('&RotFactor'),setRotFactor),
        (_('&PanFactor'),setPanFactor),
        (_('&ZoomFactor'),setZoomFactor),
        (_('&AutoZoomFactor'),setAutoZoomFactor),
        (_('&ZoomActions'),setZoomActions),
        (_('&Show Focus'),setShowFocus),
        (_('&Render Mode'),setRenderMode),
        (_('&Rendering'),setRender),
        (_('&Light0'),setLight0),
        (_('&Light1'),setLight1),
        (_('&Splash Image'),setSplash),
        (_('&Autoload Plugins'),setPlugins),
        (_('&Startup Scripts'),setAutoRun),
        (_('&Script Paths'),setScriptDirs),
        (_('&Commands'),setCommands),
        (_('&SysPath'),setSysPath),
        (_('&Help'),setHelp),
        ('---',None),
##         (_('&Edit Preferences'),editPreferences),
        (_('&Save Preferences'),savePreferences),
        (_('Toggle Timeout'),draw.timeout),
        (_('Toggle CoordsBox'),coordsbox),
        ]),
    ]


   
# End
