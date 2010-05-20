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
from pyformex.main import savePreferences
import os

from gettext import gettext as _
import utils
import widgets
from widgets import simpleInputItem as I
from widgets import groupInputItem as G
from widgets import tabInputItem as T
from widgets import compatInputItem as C
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
    GD.debug("Accepted settings:",res)
    if save is None:
        save = res.get('Save changes',None)
    if save is None:
        save = draw.ack("Save the current changes to your configuration file?")

    # Do not use 'GD.cfg.update(res)' here!
    # It will not work with our Config class!

    for k in res:
        if k.startswith('_'): # skip temporary variables
            continue
        
        if GD.cfg[k] != res[k]:
            GD.cfg[k] = res[k]
            
            if save and GD.prefcfg[k] != GD.cfg[k]:
                GD.prefcfg[k] = GD.cfg[k]

            if GD.GUI:
                if k in _activate_settings:
                    _activate_settings[k]()

    GD.debug("New settings:",GD.cfg)
    GD.debug("New preferences:",GD.prefcfg)
    #print res['render/light0']
    #print GD.cfg['render/light0']
    #print GD.prefcfg['render/light0']


def settings():
    import plugins
    
    dia = None

    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = dia.results
        res['Save changes'] = save
        GD.debug(res)
        ok_plugins = utils.subDict(res,'_plugins/')
        res['gui/plugins'] = [ p for p in ok_plugins if ok_plugins[p]]
        updateSettings(res)
        plugins.loadConfiguredPlugins()

    def acceptAndSave():
        accept(save=True)

    def autoSettings(keylist):
        return [I(k,GD.cfg[k]) for k in keylist]

    def changeScriptDirs():
        setScriptDirs()
        print "SCRIPTDIRS NOW " % GD.cfg['scriptdirs']
        dia.updateData({'scriptdirs':GD.cfg['scriptdirs']})

    def changeSplash():
        setSplash()
        dia.updateData({'gui/splash':GD.cfg['gui/splash']})


    mouse_settings = autoSettings(['gui/rotfactor','gui/panfactor','gui/zoomfactor','gui/autozoomfactor','gui/dynazoom','gui/wheelzoom'])

    plugin_items = [ I('_plugins/'+name,name in GD.cfg['gui/plugins'],text=label) for (label,name) in plugins.plugin_menus ]
    #print plugin_items

    appearence = [
        I('gui/style',GD.GUI.currentStyle(),choices=GD.GUI.getStyles()),
        I('gui/font',GD.app.font().toString(),'font'),
        ]

    cur = GD.cfg['gui/splash']
    if not cur:
        cur = GD.cfg.get('icondir','.')
    w = widgets.ImageViewerDialog(path=cur)

    dia = widgets.NewInputDialog(
        caption='pyFormex Settings',
        store=GD.cfg,
        items=[
            T('General',[
                I('syspath'),
                I('editor'),
                I('viewer'),
                I('browser'),
                I('help/docs'),
                I('autorun',text='Startup script',tooltip='This script will automatically be run at pyFormex startup'),
                I('scriptdirs',text='Script Paths',tooltip='pyFormex will look for scripts in these directories',buttons=[('Edit',changeScriptDirs)]),
                ],
             ),
            T('GUI',[
                G('Appearence',appearence),
                G('Components',[
                    I('gui/coordsbox',GD.cfg['gui/coordsbox']),
                    I('gui/showfocus',GD.cfg['gui/showfocus']),
                    I('gui/timeoutbutton',GD.cfg['gui/timeoutbutton']),
                    I('gui/timeoutvalue',GD.cfg['gui/timeoutvalue']),
                    ],
                 ),
                I('Splash image',GD.cfg['gui/splash'],buttons=[('Change',changeSplash)]),
                w,
                ],
             ),
            T('Mouse',mouse_settings),
            T('Plugins',plugin_items),
            ],
        actions=[
            ('Close',close),
            ('Accept and Save',acceptAndSave),
            ('Accept',accept),
        ])
    dia.resize(800,400)
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
    If no store is specified, the global config GD.cfg is used.
    """
    if store is None:
        store = GD.cfg
    if prefix:
        items = [ '%s/%s' % (prefix,i) for i in items ]
    itemlist = [ [ i,store[i] ] for i in items ]
    res = widgets.InputDialog(itemlist+[('Save changes',True)],'Config Dialog',GD.GUI).getResult()
    print res
    if res and store==GD.cfg:
        updateSettings(res)
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
    GD.prefcfg['pick/size'] = (int(res['w']),int(res['h']))
        
    
def setRenderMode():
    from canvas import Canvas
    res = draw.askItems([('render/mode',None,'vradio',{'text':'Render Mode','choices':Canvas.rendermodes})])
    if res:
        rendermode = res['render/mode']
        if hasattr(draw,rendermode):
            getattr(draw,rendermode)()
        updateSettings(res)
            

def set_mat_value(field):
    key = field.text()
    val = field.value()
    print key,val
    draw.set_material_value(key,val)


def set_light_value(field):
    light = field.data
    key = field.text()
    val = field.value()
    print light,key,val
    draw.set_light_value(light,key,val)
    

def createLightDialogItems(light=0):
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    tgt = 'render/light%s'%light
    val = GD.cfg[tgt]
    print "LIGHT %s" % light
    print "CFG %s " % val
    print "DICT %s" % GD.canvas.lights.lights[light].__dict__
    print "DICT %s" % dir(GD.canvas.lights.lights[light])
    
    items = [
        { 'name': 'enabled','text':'enabled' } ] + [
        I(name=k,text=k,itemtype='slider',min=0,max=100,scale=0.01,func=set_light_value,data=light)  for k in [ 'ambient', 'diffuse',  'specular' ] ] + [
        {'name': 'position','text':'position'}
        ]
    return items


def showLighting():
    print "ACCORDING TO CANVAS:"
    print GD.canvas.lights
    print "ACCORDING TO CFG:"
    print GD.cfg['render']


def setLighting():
    mat_items = [
        {'name':a,'text':a,'value':getattr(GD.canvas,a),'itemtype':'slider','min':0,'max':100,'scale':0.01,'func':set_mat_value} for a in [ 'ambient', 'specular', 'emission'] ] + [
        {'name':a,'text':a,'value':getattr(GD.canvas,a),'itemtype':'slider','min':0,'max':128,'scale':1.,'func':set_mat_value} for a in ['shininess'] ]

    enabled = [ GD.cfg['render/light%s'%light] is not None and GD.cfg['render/light%s'%light]['enabled']  for light in range(8) ]
    print "ENABLED LIGHTS"

    #items = [ ('material',mat_items) ] + [ ('light%s'%light, createLightDialogItems(light)) for light in range(8) if enabled[light]]

    choices = GD.canvas.light_model.keys()
    # DO NOT ALLOW THE LIGHT MODEL TO BE CHANGED
    choices = [ 'ambient and diffuse' ]
    items = [ {'name':'lightmodel','value':GD.canvas.lightmodel,'choices':choices}, ('material',mat_items) ] + [ ('light%s'%light, createLightDialogItems(light)) for light in range(8) if enabled[light]]
    #print items

    dia = None
    
    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = dia.results
        GD.debug(res)
        GD.cfg['render/lightmodel'] = res['render/lightmodel']
        GD.canvas.resetLighting()
        GD.app.processEvents()
        mt = utils.subDict(res,'render/material/')
        l0 = utils.subDict(res,'render/light0/')
        res = dict([ i for i in res.items() if not (i[0].startswith('render/material/') or  i[0].startswith('render/light0/'))])
        res['Save changes'] = save
        res['render/material'] = mt
        res['render/light0'] =l0
        updateSettings(res)

    def acceptAndSave():
        accept(save=True)

    dia = widgets.NewInputDialog(
        caption='pyFormex Settings',
        store=GD.cfg,
        items=items,
        prefix='render/',
        autoprefix=True,
        actions=[
            ('Close',close),
            ('Accept and Save',acceptAndSave),
            ('Apply',accept),
            ]
        )
    dia.show()
    #if res:
    #    updateSettings({tgt:res})
    #    GD.canvas.resetLights()

 
def setSplash():
    """Open an image file and set it as the splash screen."""
    cur = GD.cfg['gui/splash']
    if not cur:
        cur = GD.cfg.get('icondir','.')
    w = widgets.ImageViewerDialog(path=cur)
    fn = w.getFilename()
    w.close()
    if fn:
        GD.prefcfg['gui/splash'] = fn


def setScriptDirs():
    dia = createScriptDirsDialog()
    dia.exec_()

    
def createScriptDirsDialog():
    _dia=None
    _table=None

    def insertRow():
        ww = widgets.FileSelection(GD.cfg['workdir'],'*',exist=True,dir=True)
        fn = ww.getFilename()
        if fn:
            scr = GD.cfg['scriptdirs']
            _table.model().insertRows()
            scr[-1] = ['New',fn]
        _table.update()

    def editRow():
        row = _table.currentIndex().row()
        scr = GD.cfg['scriptdirs']
        item = scr[row]
        res = draw.askItems([('Label',item[0]),('Path',item[1])])
        if res:
            scr[row] = [res['Label'],res['Path']]
        _table.update()

    def removeRow():
        row = _table.currentIndex().row()
        _table.model().removeRows(row,1)
        _table.update()

    def moveUp():
        row = _table.currentIndex().row()
        scr = GD.cfg['scriptdirs']
        if row > 0:
            a,b = scr[row-1:row+1]
            scr[row-1] = b
            scr[row] = a
        _table.setFocus() # For some unkown reason, this seems needed to
                           # immediately update the widget
        _table.update()
        GD.app.processEvents()

    def saveTable():
        print GD.cfg['scriptdirs']
        GD.prefcfg['scriptdirs'] = GD.cfg['scriptdirs']

    #global _dia,_table
    from scriptMenu import reloadScriptMenu
    scr = GD.cfg['scriptdirs']
    _table = widgets.Table(scr,chead=['Label','Path'])
    _dia = widgets.Dialog(
        widgets=[_table],
        title='Script paths',
        actions=[('New',insertRow),('Edit',editRow),('Delete',removeRow),('Move Up',moveUp),('Reload',reloadScriptMenu),('Save',saveTable),('OK',)],
        )
    
    return _dia
        

def setOptions():
    options = ['test','uselib','safelib','fastencode']
    options = [ o for o in options if hasattr(GD.options,o) ]
    items = [ (o,getattr(GD.options,o)) for o in options ]
    debug_levels = [ 'All','None' ]
    items.append(('debug',None,'vradio',debug_levels))
    res = draw.askItems(items)
    if res:
        for o in options:
            setattr(GD.options,o,res[o])
            setattr(GD.options,'debug',debug_levels.index(res['debug'])-1)
            print(GD.options)
            ## if o == 'debug':
            ##     GD.setDebugFunc()
    


# Functions defined to delay binding
def coordsbox():
    """Toggle the coordinate display box onor off"""
    GD.GUI.coordsbox.setVisible(GD.cfg['gui/coordsbox'])
    
def timeoutbutton():
    """Toggle the timeout button on or off"""
    toolbar.addTimeoutButton(GD.GUI.toolbar)

def updateCanvas():
    GD.canvas.update()

def updateStyle():
    GD.GUI.setAppearence()

    
# This sets the functions that should be called when a setting has changed
_activate_settings = {
    'gui/coordsbox':coordsbox,
    'gui/timeoutbutton':timeoutbutton,
    'gui/showfocus':updateCanvas,
    'gui/style':updateStyle,
    'gui/font':updateStyle,
    }
   

MenuData = [
    (_('&Settings'),[
        (_('&Settings Dialog'),settings), 
        (_('&Options'),setOptions),
        ('---',None),
        (_('&Toolbar Placement'),setToolbarPlacement), 
        (_('&Draw Wait Time'),setDrawWait), 
        (_('Avg&Normal Treshold'),setAvgNormalTreshold), 
        (_('Avg&Normal Size'),setAvgNormalSize), 
        (_('&Pick Size'),setPickSize), 
        (_('&Render Mode'),setRenderMode),
        (_('&Lighting'),setLighting),
        (_('&Show Lighting'),showLighting),
        ('---',None),
        (_('&Save Preferences Now'),savePreferences),
#        (_('&Make current settings the defaults'),savePreferences),
#        (_('&Reset current settings to the saved defaults'),savePreferences),
        ]),
    ]


   
# End
