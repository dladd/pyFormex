# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
from main import savePreferences
import os

from gettext import gettext as _
import utils
import widgets
from widgets import simpleInputItem as _I, groupInputItem as _G, tabInputItem as _T
import toolbar
import draw


def updateSettings(res,save=None):
    """Update the current settings (store) with the values in res.

    res is a dictionary with configuration values.
    The current settings will be updated with the values in res.

    If res contains a key '_save_', or a `save` argument is supplied,
    and its value is True, the preferences will also be saved to the
    user's preference file.
    Else, the user will be asked whether he wants to save the changes.
    """
    pf.debug(res,"\nACCEPTED SETTINGS")
    if save is None:
        save = res.get('_save_',None)
    if save is None:
        save = draw.ack("Save the current changes to your configuration file?")

    # Do not use 'pf.cfg.update(res)' here!
    # It will not work with our Config class!

    # a set to uniquely register updatefunctions
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
        #print pf.cfg.keys()
        if pf.cfg[k] != res[k]:
            pf.cfg[k] = res[k]
            changed = True

        if changed and pf.GUI:
            # register the corresponding update function
            if k in _activate_settings:
                todo.add(_activate_settings[k])

    # We test for pf.GUI in case we want to call updateSettings before
    # the GUI is created
    if pf.GUI:
        #print todo
        for f in todo:
            f()

    pf.debug(pf.cfg,"\nNEW SETTINGS")
    pf.debug(pf.prefcfg,"\n\nNEW PREFERENCES")


def settings():
    import plugins
    import sendmail
    from elements import elementTypes
    
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
        return [_I(k,pf.cfg[k]) for k in keylist]

    def changeDirs(dircfg):
        """dircfg is a config variable that is a list of dirs"""
        setDirs(dircfg)
        dia.updateData({dircfg:pf.cfg[dircfg]})

    def changeScriptDirs():
        changeDirs('scriptdirs')
    def changeAppDirs():
        changeDirs('appdirs')


    mouse_settings = autoSettings(['gui/rotfactor','gui/panfactor','gui/zoomfactor','gui/autozoomfactor','gui/dynazoom','gui/wheelzoom'])
 
    plugin_items = [ _I('_plugins/'+name,name in pf.cfg['gui/plugins'],text=text) for name,text in plugins.pluginMenus() ]
    #print plugin_items

    appearance = [
        _I('gui/style',pf.GUI.currentStyle(),choices=pf.GUI.getStyles()),
        _I('gui/font',pf.app.font().toString(),'font'),
        ]

    toolbartip = "Currently, changing the toolbar position will only be in effect when you restart pyFormex"
    toolbars = [
        _I('gui/%s'%t,pf.cfg['gui/%s'%t],text=getattr(pf.GUI,t).windowTitle(),choices=['left','right','top','bottom'],tooltip=toolbartip) for t in [ 'camerabar','modebar','viewbar' ]
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
        _I('mail/sender',pf.cfg.get('mail/sender',sendmail.mail),text="My mail address"),
        _I('mail/server',pf.cfg.get('mail/server','localhost'),text="Outgoing mail server")
        ]

    dia = widgets.InputDialog(
        caption='pyFormex Settings',
        store=pf.cfg,
        items=[
            _T('General',[
                _I('syspath',tooltip="If you need to import modules from a non-standard path, you can supply additional paths to search here."),
                _I('editor',tooltip="The command to be used to edit a script file. The command will be executed with the path to the script file as argument."),
                _I('viewer',tooltip="The command to be used to view an HTML file. The command will be executed with the path to the HTML file as argument."),
                _I('browser',tooltip="The command to be used to browse the internet. The command will be executed with an URL as argument."),
                _I('help/docs'),
                _I('autorun',text='Startup script',tooltip='This script will automatically be run at pyFormex startup'),
                _I('scriptdirs',text='Script Paths',tooltip='pyFormex will look for scripts in these directories',buttons=[('Edit',changeScriptDirs)]),
                _I('appdirs',text='Applicationt Paths',tooltip='pyFormex will look for applications in these directories',buttons=[('Edit',changeAppDirs)]),
                _I('autoglobals',text='Auto Globals',tooltip='If checked, global Application variables of any Geometry type will automatically be copied to the pyFormex global variable dictionary (PF), and thus become available in the GUI'),
                ],
             ),
            _T('GUI',[
                _G('Appearance',appearance),
                _G('Components',toolbars+[
                    _I('gui/rerunbutton',pf.cfg['gui/rerunbutton']),
                    _I('gui/stepbutton',pf.cfg['gui/stepbutton']),
                    _I('gui/coordsbox',pf.cfg['gui/coordsbox']),
                    _I('gui/showfocus',pf.cfg['gui/showfocus']),
                    _I('gui/timeoutbutton',pf.cfg['gui/timeoutbutton']),
                    _I('gui/timeoutvalue',pf.cfg['gui/timeoutvalue']),
                    ],
                 ),
                _I('gui/splash',text='Splash image',itemtype='button',func=changeSplash),
                viewer,
                ]),
            _T('Canvas',[
                _I('_not_active_','The canvas background settings can be set from the Viewport Menu',itemtype='info',label=''),
               # _I('canvas/bgcolor',pf.cfg['canvas/bgcolor'],itemtype='color'),
                ],
              ),
            _T('Drawing',[
                _I('_info_00_',itemtype='info',text='Changes to these options currently only become effective after restarting pyFormex!'),
                _I('draw/quadline',text='Draw as quadratic lines',itemtype='list',check=True,choices=elementTypes(1),tooltip='Line elements checked here will be drawn as quadratic lines whenever possible.'),
                _I('draw/quadsurf',text='Draw as quadratic surfaces',itemtype='list',check=True,choices=elementTypes(2)+elementTypes(3),tooltip='Surface and volume elements checked here will be drawn as quadratic surfaces whenever possible.'),
                ]
              ),
            _T('Mouse',mouse_settings),
            _T('Plugins',plugin_items),
            _T('Environment',[
                _G('Mail',mail_settings),
#                _G('Jobs',jobs_settings),
                ]),
            ],
        enablers =[
            ],
        actions=[
            ('Close',close),
            ('Accept and Save',acceptAndSave),
            ('Accept',accept),
        ],
        )
    #dia.resize(800,400)
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
    itemlist = [ _I(i,store[i]) for i in items ] + [
        _I('_save_',True,text='Save changes')
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
    key = field.text().replace('material/','')
    val = field.value()
    vp = pf.GUI.viewports.current
    mat = vp.material
    mat.setValues(**{key:val})
    #print vp.material
    vp.update()

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
        _I('enabled',enabled),
        ] + [
        _I(k,val[k],itemtype='slider',min=0,max=100,scale=0.01,func=set_light_value,data=light)  for k in [ 'ambient', 'diffuse',  'specular' ]
        ] + [
        _I('position',val['position']),
        ]
    return items


def setRendering():
    import canvas

    vp = pf.GUI.viewports.current
    dia = None

    def enableLightParams(mode):
        if dia is None:
            return
        mode = str(mode)
        on = mode.startswith('smooth')
        for f in ['ambient','material']:
            dia['render/'+f].setEnabled(on)
        dia['material'].setEnabled(on)

    def updateLightParams(matname):
        matname=str(matname)
        mat = pf.GUI.materials[matname]
        val = utils.prefixDict(mat.dict(),'material/')
        print "UPDATE",val
        dia.updateData(val)
    
    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        print "RESULTS",dia.results
        if dia.results['render/mode'].startswith('smooth'):
            res = utils.subDict(dia.results,'render/',strip=False)
            matname = dia.results['render/material']
            matdata = utils.subDict(dia.results,'material/')
            # Currently, set both in cfg and Material db
            pf.cfg['material/%s' % matname] = matdata
            pf.GUI.materials[matname] = canvas.Material(matname,**matdata)
        else:
            res = utils.selectDict(dia.results,['render/mode','render/lighting'])
        res['_save_'] = save
        print "RES",res
        updateSettings(res)
        print pf.cfg
        vp = pf.GUI.viewports.current
        vp.resetLighting()
        #if pf.cfg['render/mode'] != vp.rendermode:
        print "SETMODE %s %s" % (pf.cfg['render/mode'],pf.cfg['render/lighting'])
        vp.setRenderMode(pf.cfg['render/mode'],pf.cfg['render/lighting'])
        print vp.rendermode,vp.lighting
        vp.update()
        toolbar.updateLightButton()
        

    def acceptAndSave():
        accept(save=True)
        
    def createDialog():  
        matnames = pf.GUI.materials.keys()
        mat = vp.material
        mat_items = [
            _I(a,text=a,value=getattr(mat,a),itemtype='slider',min=0,max=100,scale=0.01,func=set_mat_value) for a in [ 'ambient', 'diffuse', 'specular', 'emission']
            ] + [
            _I(a,text=a,value=getattr(mat,a),itemtype='slider',min=1,max=128,scale=1.,func=set_mat_value) for a in ['shininess']
            ]
        items = [
            _I('render/mode',vp.rendermode,text='Rendering Mode',itemtype='select',choices=canvas.Canvas.rendermodes),#,onselect=enableLightParams),
            _I('render/lighting',vp.lighting,text='Use Lighting'),
            _I('render/ambient',vp.lightprof.ambient,text='Global Ambient Lighting'),
            _I('render/material',vp.material.name,text='Material',choices=matnames,onselect=updateLightParams),
            _G('material',text='Material Parameters',items=mat_items),
            ]

        enablers = [
            ('render/lighting',True,'render/ambient','render/material','material'),
            ]
        dia = widgets.InputDialog(
            caption='pyFormex Settings',
            enablers = enablers,
            #store=pf.cfg,
            items=items,
            #prefix='render/',
            autoprefix=True,
            actions=[
                ('Close',close),
                ('Apply and Save',acceptAndSave),
                ('Apply',accept),
                ]
            )
        enableLightParams(vp.rendermode)
        return dia

    dia = createDialog()
    dia.show()


def setDirs(dircfg):
    """dircfg is a config variable that is a list of directories."""
    dia = createDirsDialog(dircfg)
    dia.exec_()
    

    
def createDirsDialog(dircfg):
    """Create a Dialog to set a list of paths.

    dircfg is a config variable that is a list of tuples (path,text)
    where path is a valid directory pathname and text is a short name
    to display in the menus.

    Examples of dircfg are 'scriptsdirs' and 'appdirs'.
    """
    
    _dia=None
    _table=None

    def insertRow():
        ww = widgets.FileSelection(pf.cfg['workdir'],'*',exist=True,dir=True)
        fn = ww.getFilename()
        if fn:
            scr = pf.cfg[dircfg]
            _table.model().insertRows()
            scr[-1] = ['New',fn]
        _table.update()

    def editRow():
        row = _table.currentIndex().row()
        scr = pf.cfg[dircfg]
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
        scr = pf.cfg[dircfg]
        if row > 0:
            a,b = scr[row-1:row+1]
            scr[row-1] = b
            scr[row] = a
        _table.setFocus() # For some unkown reason, this seems needed to
                           # immediately update the widget
        _table.update()
        pf.app.processEvents()

    def saveTable():
        pf.prefcfg[dircfg] = pf.cfg[dircfg]
        print "SAVED: %s" % pf.cfg[dircfg]

    scr = pf.cfg[dircfg]
    if dircfg == 'scriptdirs':
        title='Script paths'
        import scriptMenu
        reloadMenu = scriptMenu.reloadMenu
    else:
        title='Application paths'
        import appMenu
        reloadMenu = appMenu.reloadMenu
    _table = widgets.Table(scr,chead=['Label','Path'])
    actions=[('New',insertRow),('Edit',editRow),('Delete',removeRow),('Move Up',moveUp),('Reload',reloadMenu),('Save',saveTable),('OK',)]
        
    _dia = widgets.GenericDialog(
        widgets=[_table],
        title=title,
        actions=actions,
        )
    
    return _dia
        

def setOptions():
    options = [ 'debug' ] # Currently No user changeable options ['test']
    options = [ o for o in options if hasattr(pf.options,o) ]
    items = [ _I(o,getattr(pf.options,o)) for o in options ]
    ## # currently we only have All or None as debug levels
    ## debug_levels = [ 'All','None' ]
    ## if pf.options.debug:
    ##     debug = 'All'
    ## else:
    ##     debug = 'None'
    ## items.append(_I('debug',debug,'vradio',choices=debug_levels))
    res = draw.askItems(items)
    if res:
        print res
        for o in options:
            setattr(pf.options,o,res[o])
            #setattr(pf.options,'debug',debug_levels.index(res['debug'])-1)
            print("Options: %s" % pf.options)
            ## if o == 'debug':
            ##     pf.setDebugFunc()
    


# Functions defined to delay binding
def coordsbox():
    """Toggle the coordinate display box on or off"""
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

def updateBackground():
    #pf.canvas.setBgColor(pf.cfg['canvas/bgcolor'],pf.cfg['canvas/bgcolor2'],pf.cfg['canvas/bgmode'])
    pf.canvas.update()

def updateAppdirs():
    pf.GUI.updateAppdirs()

    
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
    'canvas/bgmode':updateBackground,
    'canvas/bgcolor':updateBackground,
    'canvas/bgcolor2':updateBackground,
    'appdirs':updateAppdirs,
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
        (_('&Rendering'),setRendering),
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
