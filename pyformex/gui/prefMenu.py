# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

import pyformex as pf
from pyformex.main import savePreferences
import os

from gettext import gettext as _
import utils
import widgets
from widgets import simpleInputItem as I, groupInputItem as G, tabInputItem as T
import toolbar
import draw


def updateSettings(res,save=None):
    """Update the current settings (store) with the values in res.

    res is a dictionary with configuration values.
    The current settings will be update with the values in res.

    If res contains a key '_save_' and its value is True, the
    preferences will also be saved to the user's preference file.
    Else, the user will be asked whether he wants to save the changes.
    """
    pf.debug(res,"\nACCEPTED SETTINGS")
    if save is None:
        save = res.get('_save_',None)
    if save is None:
        save = draw.ack("Save the current changes to your configuration file?")

    # Do not use 'pf.cfg.update(res)' here!
    # It will not work with our Config class!

    todo = set([])
    for k in res:
        if k.startswith('_'): # skip temporary variables
            continue

        changed = False
        # first try to set in prefs, as these will cascade to cfg
        if save and pf.prefcfg[k] != res[k]:
            pf.prefcfg[k] = res[k]
            changed = True
            
        # if not saved, set in cfg
        print "Setting %s = %s" % (k,res[k])
        print pf.cfg.keys()
        if pf.cfg[k] != res[k]:
            pf.cfg[k] = res[k]
            changed = True

        if changed and pf.GUI:
            if k in _activate_settings:
                todo.add(_activate_settings[k])

    # We test for pf.GUI in case we want to call updateSettings before
    # the GUI is created
    if pf.GUI:
        for f in todo:
            f()

    pf.debug(pf.cfg,"\nNEW SETTINGS")
    pf.debug(pf.prefcfg,"\n\nNEW PREFERENCES")


def settings():
    import plugins
    import sendmail
    
    dia = None

    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = dia.results
        res['_save_'] = save
        pf.debug(res)
        ok_plugins = utils.subDict(res,'_plugins/')
        res['gui/plugins'] = [ p for p in ok_plugins if ok_plugins[p]]
        updateSettings(res)
        plugins.loadConfiguredPlugins()

    def acceptAndSave():
        accept(save=True)

    def autoSettings(keylist):
        return [I(k,pf.cfg[k]) for k in keylist]

    def changeScriptDirs():
        setScriptDirs()
        pf.debug("SCRIPTDIRS NOW " % pf.cfg['scriptdirs'])
        dia.updateData({'scriptdirs':pf.cfg['scriptdirs']})


    mouse_settings = autoSettings(['gui/rotfactor','gui/panfactor','gui/zoomfactor','gui/autozoomfactor','gui/dynazoom','gui/wheelzoom'])
 
    plugin_items = [ I('_plugins/'+name,name in pf.cfg['gui/plugins'],text=text) for name,text in plugins.pluginMenus() ]
    #print plugin_items

    appearence = [
        I('gui/style',pf.GUI.currentStyle(),choices=pf.GUI.getStyles()),
        I('gui/font',pf.app.font().toString(),'font'),
        ]

    toolbartip = "Currently, changing the toolbar position will only be in effect when you restart pyFormex"
    toolbars = [
        I('gui/%s'%t,pf.cfg['gui/%s'%t],text=getattr(pf.GUI,t).windowTitle(),choices=['left','right','top','bottom'],tooltip=toolbartip) for t in [ 'camerabar','modebar','viewbar' ]
        ]

    cur = pf.cfg['gui/splash']
#    if not cur:
#        cur = pf.cfg.get('icondir','.')
    viewer = widgets.ImageView(cur,maxheight=200)

    def changeSplash(fn):
        #print "CURRENT %s" % fn
        fn = draw.askImageFile(fn)
        if fn:
            viewer.showImage(fn)
        return fn

    mail_settings = [
        I('mail/sender',pf.cfg.get('mail/sender',sendmail.mail),text="My mail address"),
        I('mail/server',pf.cfg.get('mail/server','localhost'),text="Outgoing mail server")
        ]

    dia = widgets.InputDialog(
        caption='pyFormex Settings',
        store=pf.cfg,
        items=[
            T('General',[
                I('syspath',tooltip="If you need to import modules from a non-standard path, you can supply additional paths to search here."),
                I('editor',tooltip="The command to be used to edit a script file. The command will be executed with the path to the script file as argument."),
                I('viewer',tooltip="The command to be used to view an HTML file. The command will be executed with the path to the HTML file as argument."),
                I('browser',tooltip="The command to be used to browse the internet. The command will be executed with an URL as argument."),
                I('help/docs'),
                I('autorun',text='Startup script',tooltip='This script will automatically be run at pyFormex startup'),
                I('scriptdirs',text='Script Paths',tooltip='pyFormex will look for scripts in these directories',buttons=[('Edit',changeScriptDirs)]),
                ],
             ),
            T('GUI',[
                G('Appearence',appearence),
                G('Components',toolbars+[
                    I('gui/coordsbox',pf.cfg['gui/coordsbox']),
                    I('gui/showfocus',pf.cfg['gui/showfocus']),
                    I('gui/timeoutbutton',pf.cfg['gui/timeoutbutton']),
                    I('gui/timeoutvalue',pf.cfg['gui/timeoutvalue']),
                    ],
                 ),
                I('gui/splash',text='Splash image',itemtype='button',func=changeSplash),
                viewer,
                ],
             ),
            T('Mouse',mouse_settings),
            T('Plugins',plugin_items),
            T('Environment',mail_settings),
                
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
    If no store is specified, the global config pf.cfg is used.
    """
    if store is None:
        store = pf.cfg
    if prefix:
        items = [ '%s/%s' % (prefix,i) for i in items ]
    itemlist = [ I(i,store[i]) for i in items ] + [
        I('_save_',True,text='Save changes')
        ]
    res = widgets.InputDialog(itemlist,'Config Dialog',pf.GUI).getResult()
    pf.debug(res)
    if res and store==pf.cfg:
        updateSettings(res)
    return res

 
def setDrawWait():
    askConfigPreferences(['draw/wait'])
    pf.GUI.drawwait = pf.cfg['draw/wait']

def setLinewidth():
    askConfigPreferences(['draw/linewidth'])


def setAvgNormalTreshold():
    askConfigPreferences(['render/avgnormaltreshold'])
def setAvgNormalSize():
    askConfigPreferences(['mark/avgnormalsize'])

def setSize():
    pf.GUI.resize(800,600)

def setPickSize():
    w,h = pf.cfg['pick/size']
    res = draw.askItems([['w',w],['h',h]])
    pf.prefcfg['pick/size'] = (int(res['w']),int(res['h']))
            

def set_mat_value(field):
    key = field.text()
    val = field.value()
    mat = pf.GUI.viewports.current
    mat.setValues({key:val})


def set_light_value(field):
    light = field.data
    key = field.text()
    val = field.value()
    #print light,key,val
    draw.set_light_value(light,key,val)
    

def createLightDialogItems(light=0,enabled=True):
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    tgt = 'render/light%s'%light
    val = pf.cfg[tgt]
    print "LIGHT %s" % light
    print "CFG %s " % val
    #print "DICT %s" % pf.canvas.lights.lights[light].__dict__
    #print "DICT %s" % dir(pf.canvas.lights.lights[light])
    
    items = [
        I('enabled',enabled),
        ] + [
        I(k,val[k],itemtype='slider',min=0,max=100,scale=0.01,func=set_light_value,data=light)  for k in [ 'ambient', 'diffuse',  'specular' ]
        ] + [
        I('position',val['position']),
        ]
    return items


def showLighting():
    print "ACCORDING TO CANVAS:"
    print pf.canvas.lights
    print "ACCORDING TO CFG:"
    print pf.cfg['render']


## def setLighting():
##     nlights = pf.cfg['render/nlights']
##     mat_items = [
##         {'name':a,'text':a,'value':getattr(pf.canvas,a),'itemtype':'slider','min':0,'max':100,'scale':0.01,'func':set_mat_value } for a in [ 'ambient', 'diffuse', 'specular', 'emission'] ] + [
##         {'name':a,'text':a,'value':getattr(pf.canvas,a),'itemtype':'slider','min':1,'max':64,'scale':1.,'func':set_mat_value } for a in ['shininess'] ]

##     enabled = [ pf.cfg['render/light%s'%light] is not None and pf.cfg['render/light%s'%light].get('enabled',False) for light in range(nlights) ]
##     pf.debug("ENABLED LIGHTS")

##     choices = pf.canvas.light_model.keys()
##     # DO NOT ALLOW THE LIGHT MODEL TO BE CHANGED
##     choices = [ 'ambient and diffuse' ]
##     items = [
##         {'name':'lightmodel','value':pf.canvas.lightmodel,'choices':choices,'tooltip':"""The light model defines which light components are set by the color setting functions. The default light model is 'ambient and diffuse'. The other modes are experimentally. Use them only if you know what you are doing."""},
##         G('material',mat_items),
## #        I('nlights',4,text='Number of lights'),
##         ] + [
##         T('light%s'%light, createLightDialogItems(light,True)) for light in range(nlights)
##         ]

##     enablers = [
##         ('lightmodel','','material/ambient','material/diffuse'),
##         ]

##     dia = None
    
##     def close():
##         dia.close()
        
##     def accept(save=False):
##         dia.acceptData()
##         #print "RESULTS",dia.results
##         res = {}
##         res['material'] = utils.subDict(dia.results,'material/')
##         for i in range(8):
##             key = 'light%s'%i
##             res[key] = utils.subDict(dia.results,key+'/')
##         rest = [ k for k in dia.results.keys() if not (k.startswith('material') or  k.startswith('light')) ]
##         rest = dict((k,dia.results[k]) for k in rest)
##         res.update(rest)
##         res = utils.prefixDict(res,'render/')
##         res['_save_'] = save
##         updateSettings(res)

##     def acceptAndSave():
##         accept(save=True)

##     def addLight():
##         accept(save=False)
##         dia.close()
        
##     def createDialog():  
##         dia = widgets.InputDialog(
##             caption='pyFormex Settings',
##             enablers = enablers,
##             #store=pf.cfg,
##             items=items,
##             prefix='render/',
##             autoprefix=True,
##             actions=[
##                 ('Close',close),
##                 ('Accept and Save',acceptAndSave),
##                 ('Apply',accept),
##                 ]
##             )
##         return dia

##     dia = createDialog()
##     dia.show()
##     #if res:
##     #    updateSettings({tgt:res})
##     #    pf.canvas.resetLights()


def setRenderMode():
    import canvas

    vp = pf.GUI.viewports.current
    dia = None

    def enableLightParams(mode):
        print "enableLightParams %s" % dia
        if dia is None:
            return
        mode = str(mode)
        on = mode.startswith('smooth')
        print "ON = %s" % on
        for f in ['ambient','material']:
            print dia['render/'+f]
            dia['render/'+f].setEnabled(on)
        
    
    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = dia.results
        res['_save_'] = save
        print res
        updateSettings(res)
        print pf.cfg
        vp.resetLighting()
        if pf.cfg['render/mode'] != vp.rendermode:
            vp.setRenderMode(pf.cfg['render/mode'])
        #print "VP"
        #print pf.cfg['render/mode'],pf.cfg['render/ambient'],pf.cfg['render/material']
        #print vp.rendermode,vp.lightprof.ambient,vp.material.name
        #vp.setRenderMode(res['mode'])
        #vp.setAmbient(res['ambient'])
        #vp.setMaterial(res['material'])
        #print "AGAIN",vp.rendermode,vp.lightprof.ambient,vp.material.name
        vp.update()

    def acceptAndSave():
        accept(save=True)
        
    def createDialog():  
        matnames = pf.GUI.materials.keys()
        items = [
            I('mode',vp.rendermode,text='Rendering Mode',itemtype='select',choices=canvas.Canvas.rendermodes,onselect=enableLightParams),
            I('ambient',vp.lightprof.ambient,text='Global Ambient Lighting'),
            I('material',vp.material.name,text='Material',choices=matnames),
            ]
        dia = widgets.InputDialog(
            caption='pyFormex Settings',
            #enablers = enablers,
            #store=pf.cfg,
            items=items,
            prefix='render/',
            #autoprefix=True,
            actions=[
                ('Close',close),
                ('Accept and Save',acceptAndSave),
                ('Apply',accept),
                ]
            )
        enableLightParams(vp.rendermode)
        return dia

    dia = createDialog()
    dia.show()


def changeMaterial():
    vp = pf.GUI.viewports.current
    mats = getMaterials()
    matnames = mats.keys()
    matname = vp.matname
    mat = vp.material
    print mat
    items = [
        I('material',pf.cfg['render/material'],choices=matnames),
        ] + [
        {'name':a,'text':a,'value':mat[a],'itemtype':'slider','min':0,'max':100,'scale':0.01,'func':set_mat_value } for a in [ 'ambient', 'diffuse', 'specular', 'emission']
        ] + [
        {'name':a,'text':a,'value':mat[a],'itemtype':'slider','min':1,'max':64,'scale':1.,'func':set_mat_value } for a in ['shininess']
        ]

    dia = None
    
    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = {}
        res['render/material'] = dia.results['default']
        res['_save_'] = save
        print res
        #updateSettings(res)
        vp.setMaterial(dia.results['current'])
        vp.update()

    def acceptAndSave():
        accept(save=True)
        
    def createDialog():  
        dia = widgets.InputDialog(
            caption='pyFormex Settings',
            #enablers = enablers,
            #store=pf.cfg,
            items=items,
            #prefix='render/',
            #autoprefix=True,
            actions=[
                ('Close',close),
                ('Accept and Save',acceptAndSave),
                ('Apply',accept),
                ]
            )
        return dia

    dia = createDialog()
    dia.show()
        


def setScriptDirs():
    dia = createScriptDirsDialog()
    dia.exec_()

    
def createScriptDirsDialog():
    _dia=None
    _table=None

    def insertRow():
        ww = widgets.FileSelection(pf.cfg['workdir'],'*',exist=True,dir=True)
        fn = ww.getFilename()
        if fn:
            scr = pf.cfg['scriptdirs']
            _table.model().insertRows()
            scr[-1] = ['New',fn]
        _table.update()

    def editRow():
        row = _table.currentIndex().row()
        scr = pf.cfg['scriptdirs']
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
        scr = pf.cfg['scriptdirs']
        if row > 0:
            a,b = scr[row-1:row+1]
            scr[row-1] = b
            scr[row] = a
        _table.setFocus() # For some unkown reason, this seems needed to
                           # immediately update the widget
        _table.update()
        pf.app.processEvents()

    def saveTable():
        #print pf.cfg['scriptdirs']
        pf.prefcfg['scriptdirs'] = pf.cfg['scriptdirs']

    #global _dia,_table
    from scriptMenu import reloadScriptMenu
    scr = pf.cfg['scriptdirs']
    _table = widgets.Table(scr,chead=['Label','Path'])
    _dia = widgets.GenericDialog(
        widgets=[_table],
        title='Script paths',
        actions=[('New',insertRow),('Edit',editRow),('Delete',removeRow),('Move Up',moveUp),('Reload',reloadScriptMenu),('Save',saveTable),('OK',)],
        )
    
    return _dia
        

def setOptions():
    options = ['test','uselib','safelib','fastencode']
    options = [ o for o in options if hasattr(pf.options,o) ]
    items = [ I(o,getattr(pf.options,o)) for o in options ]
    # currently we only have All or None as debug levels
    debug_levels = [ 'All','None' ]
    if pf.options.debug:
        debug = 'All'
    else:
        debug = 'None'
    items.append(I('debug',debug,'vradio',choices=debug_levels))
    res = draw.askItems(items)
    if res:
        for o in options:
            setattr(pf.options,o,res[o])
            setattr(pf.options,'debug',debug_levels.index(res['debug'])-1)
            print("Options: %s" % pf.options)
            ## if o == 'debug':
            ##     pf.setDebugFunc()
    


# Functions defined to delay binding
def coordsbox():
    """Toggle the coordinate display box onor off"""
    pf.GUI.coordsbox.setVisible(pf.cfg['gui/coordsbox'])
    
def timeoutbutton():
    """Toggle the timeout button on or off"""
    toolbar.addTimeoutButton(pf.GUI.toolbar)

def updateCanvas():
    pf.canvas.update()

def updateStyle():
    pf.GUI.setAppearence()
 

def updateToolbars():
    pf.GUI.updateToolBars()

    
# This sets the functions that should be called when a setting has changed
_activate_settings = {
    'gui/coordsbox':coordsbox,
    'gui/timeoutbutton':timeoutbutton,
    'gui/showfocus':updateCanvas,
    'gui/style':updateStyle,
    'gui/font':updateStyle,
    'gui/camerabar':updateToolbars,
    'gui/viewbar':updateToolbars,
    'gui/modebar':updateToolbars,
    }
   

MenuData = [
    (_('&Settings'),[
        (_('&Settings Dialog'),settings), 
        (_('&Options'),setOptions),
        ('---',None),
        (_('&Draw Wait Time'),setDrawWait), 
        (_('Avg&Normal Treshold'),setAvgNormalTreshold), 
        (_('Avg&Normal Size'),setAvgNormalSize), 
        (_('&Pick Size'),setPickSize), 
        (_('&Rendering'),setRenderMode),
        #(_('&Set Material Type'),setMaterial),
        #(_('&Change Material Parameters'),changeMaterial),
        ## (_('&Show Lighting'),showLighting),
        ('---',None),
        (_('&Save Preferences Now'),savePreferences),
#        (_('&Make current settings the defaults'),savePreferences),
#        (_('&Reset current settings to the saved defaults'),savePreferences),
        ]),
    ]


   
# End
