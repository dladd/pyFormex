# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
"""Functions for the Pref menu.

"""
from __future__ import print_function

import pyformex as pf
from main import savePreferences
import os

from gettext import gettext as _
import utils
import widgets
from widgets import simpleInputItem as _I, groupInputItem as _G, tabInputItem as _T
import toolbar
import draw
import canvas


def updateSettings(res,save=None):
    """Update the current settings (store) with the values in res.

    res is a dictionary with configuration values.
    The current settings will be updated with the values in res.

    If res contains a key '_save_', or a `save` argument is supplied,
    and its value is True, the preferences will also be saved to the
    user's preference file.
    Else, the user will be asked whether he wants to save the changes.
    """
    pf.debug("\nACCEPTED SETTINGS\n%s"% res,pf.DEBUG.CONFIG)
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
        print("Setting %s = %s" % (k,res[k]))
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
        for f in todo:
            f()

    pf.debug("\nNEW SETTINGS\n%s"%pf.cfg,pf.DEBUG.CONFIG)
    pf.debug("\nNEW PREFERENCES\n%s"%pf.prefcfg,pf.DEBUG.CONFIG)


def settings():
    """Interactively change the pyformex settings.

    Creates a dialog to change (most of) the pyformex user configuration.
    To change the canvas setttings, use viewportMenu.canvasSettings.
    """
    import plugins
    import sendmail
    from elements import elementTypes

    dia = None
    _actionbuttons = [ 'play', 'rerun', 'step', 'continue', 'stop', 'edit', 'info' ]

    def close():
        dia.close()

    def accept(save=False):
        dia.acceptData()
        res = dia.results
        res['_save_'] = save
        ok_plugins = utils.subDict(res,'_plugins/')
        res['gui/plugins'] = [ p for p in ok_plugins if ok_plugins[p]]
        res['gui/actionbuttons'] = [ t for t in _actionbuttons if res['_gui/%sbutton'%t ] ]
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

    # Use _ to avoid adding these items in the config
    plugin_items = [ _I('_plugins/'+name,name in pf.cfg['gui/plugins'],text=text) for name,text in plugins.pluginMenus() ]

    appearance = [
        _I('gui/style',pf.app.currentStyle(),choices=pf.app.getStyles()),
        _I('gui/font',pf.app.font().toString(),'font'),
        ]

    toolbartip = "Currently, changing the toolbar position will only be in effect when you restart pyFormex"
    toolbars = [
        _I('gui/%s'%t,pf.cfg['gui/%s'%t],text=getattr(pf.GUI,t).windowTitle(),choices=['left','right','top','bottom'],tooltip=toolbartip) for t in [ 'camerabar','modebar','viewbar' ]
        ]
    # Use _ to avoid adding these items in the config
    actionbuttons = [
        _I('_gui/%sbutton'%t,t in pf.cfg['gui/actionbuttons'],text="%s Button" % t.capitalize()) for t in _actionbuttons
        ]


    cur = pf.cfg['gui/splash']
#    if not cur:
#        cur = pf.cfg.get('icondir','.')
    viewer = widgets.ImageView(cur,maxheight=200)

    def changeSplash(fn):
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
                _I('showapploaderrors',text='Show Application Load Error Traceback',tooltip='If checked, a traceback of exceptions occurring at Application Load time will be output. If unchecked, such exceptions will be suppressed, and the application will not run.'),
                _I('loadcurproj',text="Reload last project on startup"),
                _I('check_print',text="Check scripts for the use of the print statement"),
                _I('plot2d',text="Prefered 2D plot library",choices=['gnuplot','matplotlib']),
                _I('commands',text='Use commands module instead of subprocess',tooltip="If checked, pyFormex will use the Python 'commands' module for the execution of external commands. The default (unchecked) is to use the 'subprocess' module. If you notice a lot of command failures, or even hangups, you may want to switch."),
                ],
             ),
            _T('GUI',[
                _G('Appearance',appearance),
                _G('Components',toolbars+actionbuttons+[
                    _I('gui/coordsbox'),
                    _I('gui/showfocus',pf.cfg['gui/showfocus']),
                    _I('gui/runalloption',pf.cfg['gui/runalloption']),
                    _I('gui/timeoutbutton',pf.cfg['gui/timeoutbutton']),
                    _I('gui/timeoutvalue',pf.cfg['gui/timeoutvalue']),
                    ],
                 ),
                _I('gui/splash',text='Splash image',itemtype='button',func=changeSplash),
                viewer,
                ]),
            _T('Canvas',[
                _I('_not_active_','The canvas settings can be set from the Viewport Menu',itemtype='info',text=''),
                ],
              ),
            _T('Drawing',[
                _I('draw/rendermode',pf.cfg['draw/rendermode'],choices=canvas.CanvasSettings.RenderProfiles),
                _I('draw/wait',pf.cfg['draw/wait']),
                _I('draw/picksize',pf.cfg['draw/picksize']),
                _I('draw/disable_depth_test',pf.cfg['draw/disable_depth_test'],text='Disable depth testing for transparent actors'),
                _I('render/avgnormaltreshold',pf.cfg['render/avgnormaltreshold']),
                _I('_info_00_',itemtype='info',text='Changes to the options below will only become effective after restarting pyFormex!'),
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

    This function can be used to change individual values by a simpler
    interface than the full settings dialog.
    """
    if store is None:
        store = pf.cfg
    if prefix:
        items = [ '%s/%s' % (prefix,i) for i in items ]
    itemlist = [ _I(i,store[i]) for i in items ] + [
        _I('_save_',True,text='Save changes')
        ]
    res = widgets.InputDialog(itemlist,'Config Dialog',pf.GUI).getResults()
    #pf.debug(res,pf.DEBUG.CONFIG)
    if res and store==pf.cfg:
        updateSettings(res)
    return res


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
    print("LIGHT %s" % light)
    print("CFG %s " % val)

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
        print("UPDATE",val)
        dia.updateData(val)

    def close():
        dia.close()

    def accept(save=False):
        dia.acceptData()
        print("RESULTS",dia.results)
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
        print("RES",res)
        updateSettings(res)
        print(pf.cfg)
        vp = pf.GUI.viewports.current
        vp.resetLighting()
        #if pf.cfg['render/mode'] != vp.rendermode:
        print("SETMODE %s %s" % (pf.cfg['render/mode'],pf.cfg['render/lighting']))
        vp.setRenderMode(pf.cfg['render/mode'],pf.cfg['render/lighting'])
        print(vp.rendermode,vp.settings.lighting)
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
            _I('render/mode',vp.rendermode,text='Rendering Mode',itemtype='select',choices=draw.renderModes()),#,onselect=enableLightParams),
            _I('render/lighting',vp.settings.lighting,text='Use Lighting'),
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
    """Configure the paths from which to read apps/scripts

    dircfg is a config variable that is a list of directories.
    It should be one of 'appdirs' or 'scriptdirs'.
    The config value should be a list of tuples ['text','path'].
    """
    dia = createDirsDialog(dircfg)
    dia.exec_()
    if (dia.result()==widgets.ACCEPTED):
        import appMenu
        #print(dia.results)
        data = dia.results[dircfg]
        pf.prefcfg[dircfg] = pf.cfg[dircfg] = map(tuple,data)
        #print("SET %s to %s" % (dircfg,pf.prefcfg[dircfg]))
        appMenu.reloadMenu(mode=dircfg[:-4])


def createDirsDialog(dircfg):
    """Create a Dialog to set a list of paths.

    dircfg is a config variable that is a list of directories.
    """
    _dia=None
    _table=None

    mode = dircfg[:-4]
    if mode == 'app':
        title = 'Application paths'
    else:
        title='Script paths'

    def insertRow():
        ww = widgets.FileSelection(pf.cfg['workdir'],'*',exist=True,dir=True)
        fn = ww.getFilename()
        if fn:
            _table.model().insertRows()
            _table.model()._data[-1] = [os.path.basename(fn).capitalize(),fn]
        _table.update()

    def removeRow():
        row = _table.currentIndex().row()
        _table.model().removeRows(row,1)
        _table.update()

    def moveUp():
        row = _table.currentIndex().row()
        if row > 0:
            a,b = _table.model()._data[row-1:row+1]
            _table.model()._data[row-1:row+1] = b,a
        _table.setFocus() # For some unkown reason, this seems needed to
                          # immediately update the widget
        _table.update()
        pf.app.processEvents()

    def reloadMenu():
        import appMenu
        appMenu.reloadMenu(mode=mode)

    data = map(list,pf.cfg[dircfg])
    _dia = widgets.InputDialog(
        items = [
            _I(dircfg,data,itemtype='table',chead = ['Label','Path']),
            ],
        actions = [
            ('New',insertRow),
            ('Delete',removeRow),
            ('Move Up',moveUp),
            ('OK',),
            ('Cancel',),
            ],
        caption=title)
    _table = _dia[dircfg].input


    return _dia


def setDebug():
    options = [ o for o in dir(pf.DEBUG) if o[0] != '_' ]
    options.remove('ALL')
    options.remove('NONE')
    values = [ getattr(pf.DEBUG,o) for o in options ]
    items = [ _I(o,bool(pf.options.debuglevel & v)) for o,v in zip(options,values) ]
    res = draw.askItems(items)
    if res:
        print(res)
        debug = 0
        for o,v in zip(options,values):
            if res[o]:
                debug |= v
        print("debuglevel = %s" % debug)
        pf.options.debuglevel = debug


def setOptions():
    options = [ 'redirect','debuglevel','rst2html']
    options = [ o for o in options if hasattr(pf.options,o) ]
    items = [ _I(o,getattr(pf.options,o)) for o in options ]
    res = draw.askItems(items)
    if res:
        print(res)
        for o in options:
            setattr(pf.options,o,res[o])
            print("Options: %s" % pf.options)
            if o == 'redirect':
                pf.GUI.board.redirect(pf.options.redirect)



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
    pf.canvas.update()

def updateAppdirs():
    pf.GUI.updateAppdirs()

def updateDrawWait():
    pf.GUI.drawwait = pf.cfg['draw/wait']


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
    'canvas/bgimage':updateBackground,
    'appdirs':updateAppdirs,
    'draw/wait':updateDrawWait,
    }

def setDrawWait():
    askConfigPreferences(['draw/wait'])

MenuData = [
    (_('&Settings'),[
        (_('&Settings Dialog'),settings),
        (_('&Debug'),setDebug),
        (_('&Options'),setOptions),
        (_('&Draw Wait'),setDrawWait),
        (_('&Rendering Params'),setRendering),
        ('---',None),
        (_('&Save Preferences Now'),savePreferences),
#        (_('&Make current settings the defaults'),savePreferences),
#        (_('&Reset current settings to the saved defaults'),savePreferences),
        ]),
    ]



# End
