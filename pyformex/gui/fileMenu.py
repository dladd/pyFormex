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
"""Functions from the File menu.

"""
from __future__ import print_function

import os,shutil
import pyformex as pf
import widgets
import utils
import project
import draw
from draw import _I,_G
import image
import plugins

from gettext import gettext as _
from prefMenu import updateSettings


##################### handle project files ##########################


def openProject(fn=None,exist=False,access=['wr','rw','w','r'],default=None):
    """Open a (new or old) Project file.

    A dialog is presented to ask the user for a Project file name and the
    access modalities. The parameters help in setting sensible defaults
    for the user and in delimiting his options.
    Depending on he results of the dialog, either a new project is created or
    an old one is opened, or nothing is done.
    If a project is opened, it is returned, else the return value is None.

    Parameters:

    - `fn`: filename: if specified, the Project file dialog will start with
      the specified file, otherwise it will start in the current directory.
    - `exist`: boolean: if False (default), the user can create new project
      files as well as open existing ones. Use exist=True or
      :func:`openExistingProject` to only accept existing project files.
    - `access`: a list of :class:`Project` access modes to be presented to
      the user.
    - `default`: the access mode that is presented as default to the user.
      If not specified, the first option of `access` will be the default.
    """
    if type(access) == str:
        access = [access]
    cur = fn if fn else '.'
    typ = utils.fileDescription(['pyf','all'])
    res = widgets.ProjectSelection(cur,typ,exist=exist,access=access,default=default,convert=True).getResult()
    if not res:
        return

    fn = res.fn
    if not fn.endswith('.pyf'):
        fn += '.pyf'
    access = res.acc
    compression = res.cpr
    convert = res.cvt
    signature = pf.FullVersion()

    # OK, we have all data, now create/open the project
    pf.message("Opening project %s" % fn)
    pf.GUI.setBusy() # loading  may take a while
    try:
        proj = project.Project(fn,access=access,convert=convert,signature=signature,compression=compression)
        if proj.signature != signature:
            pf.warning("The project was written with %s, while you are now running %s. If the latter is the newer one, this should probably not cause any problems. Saving is always done in the current format of the running version. Save your project and this message will be avoided on the next reopening." % (proj.signature,signature))
    except:
        proj = None
        raise
    finally:
        pf.GUI.setBusy(False)

    proj.hits = 0
    pf.debug("START COUNTING HITS",pf.DEBUG.PROJECT)
    return proj


def readProjectFile(fn):
    if os.path.exists(fn):
        proj = project.Project(fn,access='wr')
        return proj
    else:
        return None


def setProject(proj):
    """Make the specified project the current project.

    proj is an open project.
    If a filename, the project file is opened.

    .. note: The remainder is obsolete

    The user is asked for a Project file name and the access modalities.
    Depending on the results of the dialog:

    - either an new project is create or an old is opened,
    - the old data may be discarded, added to the current pyFormex globals,
      or replace them
    - the opened Project may become the current Project, or its data are
      just imported in the current Project.


    The default will let the user create new project files as well as open
    existing ones.
    Use create=False or the convenience function openProject to only accept
    existing project files.

    If a compression level (1..9) is given, the contents will be compressed,
    resulting in much smaller project files at the cost of

    Only one pyFormex project can be open at any time. The open project
    owns all the global data created and exported by any script.

    If makeDefault is True, an already open project will be closed and
    the opened project becomes the current project.
    If makeDefault is False, the project data are imported into pf.PF
    and the current project does not change. This means that if a project was
    open, the imported data will be added to it.

    If addGlobals is None, the user is asked whether the current globals
    should be added to the project. Set True or False to force or reject
    the adding without asking.
    """
    pf.message("Setting current project to %s" % proj.filename)
    pf.message("Project contents: %s" % utils.sortedKeys(proj))
    keep = {}
    if pf.PF:
        pf.message("Current pyFormex globals: %s" % utils.sortedKeys(pf.PF))
        _delete = "Delete"
        _add = "Keep non existing"
        _overwrite = "Keep all (overwrite project)"
        res = draw.ask("What shall I do with the current pyFormex globals?",[_delete,_add,_overwrite])
        if res == _add:
            keep = utils.removeDict(pf.PF,proj)
        elif res == _overwrite:
            keep = pf.PF
    pf.PF = proj
    if keep:
        pf.PF.update(keep)
    if pf.PF.filename:
        updateSettings({
            'curproj':pf.PF.filename,
            'workdir':os.path.dirname(pf.PF.filename),
            },save=True)
    pf.GUI.setcurproj(pf.PF.filename)

    if hasattr(proj,'_autoscript_'):
        _ignore = "Ignore it!"
        _show = "Show it"
        _edit = "Load it in the editor"
        _exec = "Execute it"
        res = draw.ask("There is an autoscript stored inside the project.\nIf you received this project file from an untrusted source, you should probably not execute it.",[_ignore,_show,_edit,_exec])
        if res == _show:
            res = draw.showText(proj._autoscript_)#,actions=[_ignore,_edit,_show])
            return
        if res == _exec:
            draw.playScript(proj._autoscript_)
        elif res == _edit:
            fn = "_autoscript_.py"
            draw.checkWorkdir()
            f = open(fn,'w')
            f.write(proj._autoscript_)
            f.close()
            openScript(fn)
            editApp(fn)

    if hasattr(proj,'autofile') and draw.ack("The project has an autofile attribute: %s\nShall I execute this script?" % proj.autofile):
        draw.processArgs([proj.autofile])

    listProject()


def createProject():
    """Open an new project.

    Ask the user to select an existing project file, and then open it.
    """
    closeProject()
    proj = openProject(pf.PF.filename,exist=False)
    if proj is not None: # may be empty
        setProject(proj)


def openExistingProject():
    """Open an existing project.

    Ask the user to select an existing project file, and then open it.
    """
    closeProject()
    proj = openProject(pf.PF.filename,exist=True)
    if proj is not None: # may be empty
        setProject(proj)


def importProject():
    """Import an existing project.

    Ask the user to select an existing project file, and then import
    all or selected data from it into the current project.
    """
    proj = openProject(exist=True,access='r')
    if proj: # only if non-empty
        keys = utils.sortedKeys(proj)
        res = draw.askItems(
            [   _I('mode',choices=['All','Defined','Undefined','Selected','None'],itemtype='radio'),
                _I('selected',choices=keys,itemtype='list'),
                ],
            caption='Select variables to import',
            )
        if res:
            mode = res['mode'][0]
            if mode == 'A':
                pass
            elif mode == 'D':
                proj = utils.selectDict(proj,pf.PF)
            elif mode == 'U':
                proj = utils.removeDict(proj,pf.PF)
            elif mode == 'S':
                proj = utils.selectDict(proj,res['selected'])
            elif mode == 'N':
                return
            pf.message("Importing symbols: %s" % utils.sortedKeys(proj))
            pf.PF.update(proj)
            listProject()


def setAutoScript():
    """Set the current script as autoScript in the project"""
    if pf.cfg['curfile'] and pf.GUI.canPlay:
        pf.PF._autoscript_ = open(pf.cfg['curfile']).read()


def setAutoFile():
    """Set the current script/app as autofile in the project"""
    if pf.cfg['curfile'] and pf.GUI.canPlay:
        pf.PF.autofile = pf.cfg['curfile']


def removeAutoScript():
    delattr(pf.PF,'_autoscript_')


def removeAutoFile():
    delattr(pf.PF,'autofile')


def saveProject():
    """Save the current project.

    If the current project is a named one, its contents are written to file.
    This function does nothing if the current project is a temporary one.
    """
    if pf.PF.filename is not None:
        pf.message("Saving Project contents: %s" % utils.sortedKeys(pf.PF))
        pf.GUI.setBusy()
        pf.PF.save()
        pf.GUI.setBusy(False)


def saveAsProject():
    proj = openProject(pf.PF.filename,exist=False,access=['w'],default='w')
    if proj is not None: # even if empty
        pf.PF.filename = proj.filename
        pf.PF.gzip = proj.gzip
        pf.PF.signature = proj.signature  # put new signature inside
        saveProject()
    if pf.PF.filename is not None:
        updateSettings({
            'curproj':pf.PF.filename,
            'workdir':os.path.dirname(pf.PF.filename),
            },save=True)
    pf.GUI.setcurproj(pf.PF.filename)


def listProject():
    """Print all global variable names."""
    pf.message("pyFormex globals: %s" % utils.sortedKeys(pf.PF))

def clearProject():
    """Clear the contents of the current project."""
    pf.PF.clear()

def closeProject(save=None,delet=None):
    """Close the current project, saving it or not.

    Parameters:

    - `save`: None, True or False. Determines whether the project should be
      saved prior to closing it. If None, it will be asked from the user.
      Note that this parameter is only used for named Projects. Temporary
      Projects are never saved implicitely.
    """
    if pf.PF.filename is not None:
        if save is None:
            save = draw.ack("Save the current project before closing it?")
        pf.message("Closing project %s (save=%s)" % (pf.PF.filename,save))
        if save:
            saveProject()
            if pf.PF:
                listProject()
                if delet is None:
                    delet = draw.ask("What shall I do with the existing globals?",["Delete","Keep"]) == "Delete"
                if delet:
                    pf.PF.clear()

    pf.PF.filename = None
    pf.GUI.setcurproj('None')
    updateSettings({
        'curproj':pf.PF.filename,
        },save=True)


def closeProjectWithoutSaving():
    """Close the current project without saving it."""
    closeProject(False)


def convertProjectFile():
    proj = openProject(pf.PF.filename,access=['c'],default='c',exist=True)
    if proj is not None:
        pf.debug("Converting project file %s" % proj.filename,pf.DEBUG.PROJECT|pf.DEBUG.INFO)
        proj.convert(proj.filename.replace('.pyf','_converted.pyf'))


def uncompressProjectFile():
    proj = openProject(pf.PF.filename,access=['u'],default='u',exist=True)
    if proj is not None:
        proj.uncompress(proj.filename.replace('.pyf','_uncompressed.pyf'))


##################### handle script files ##########################

def openScript(fn=None,exist=True,create=False):
    """Open a pyFormex script and set it as the current script.

    If no filename is specified, a file selection dialog is started to select
    an existing script, or allow to create a new file if exist is False.

    If the file exists and is a pyFormex script, it is set ready to execute.

    If create is True, a default pyFormex script template will be written
    to the file, overwriting the contents if the file existed. Then, the
    script is loaded into the editor.

    We encourage the use of createScript() to create new scripts and leave
    openScript() to open existing scripts.
    """
    if fn is None:
        cur = pf.cfg['curfile']
        if cur is None:
            cur = pf.cfg['workdir']
        if cur is None:
            cur  = '.'
        typ = utils.fileDescription('pyformex')
        fn = widgets.FileSelection(cur,typ,exist=exist).getFilename()
    if fn:
        if create:
            if not exist and os.path.exists(fn) and not draw.ack("The file %s already exists.\n Are you sure you want to overwrite it?" % fn):
                return None
            template = pf.cfg['scripttemplate']
            if (os.path.exists(template)):
                shutil.copyfile(template,fn)
        updateSettings({'workdir':os.path.dirname(fn)},save=True)
        pf.GUI.setcurfile(fn)
        pf.GUI.scripthistory.add(fn)
        if create:
            draw.editFile(fn)
    return fn


def createScript(fn=None):
    return openScript(fn,exist=False,create=True)


def editApp(appname=None):
    """Edit an application source file.

    If no appname is specified, the current application is used.

    This loads the application module, gets its file name, and if
    it is a source file or the the corresponding source file exists,
    that file is loaded into the editor.
    """
    if appname is None:
        appname = pf.cfg['curfile']

    if utils.is_script(appname):
        # this is a script, not an app
        fn = appname

    else:
        import apps
        app = apps.load(appname)
        if app is None:
            fn = apps.findAppSource(appname)
        else:
            fn = apps.findAppSource(app)
        if not os.path.exists(fn):
            draw.warning("The file '%s' does not exist" % fn)
            return

    draw.editFile(fn)


##################### other functions ##########################


def saveImage(multi=False):
    """Save an image to file.

    This will show the Save Image dialog, with the multisave mode checked if
    multi = True. Then, depending on the user's selection, it will either:
     - save the current Canvas/Window to file
     - start the multisave/autosave mode
     - do nothing
    """
    pat = map(utils.fileDescription, ['img','icon','all'])
    dia = widgets.SaveImageDialog(pf.cfg['workdir'],pat,multi=multi)
    opt = dia.getResult()
    if opt:
        if opt.fm == 'From Extension':
            if utils.fileTypeFromExt(opt.fn) == '':
                opt.fn += '.png'
            opt.fm = None
        if opt.qu < 0:
            opt.qu = -1
        updateSettings({'workdir':os.path.dirname(opt.fn)},save=True)
        image.save(filename=opt.fn,
                   format=opt.fm,
                   quality=opt.qu,
                   size=opt.sz,
                   window=opt.wi,
                   multi=opt.mu,
                   hotkey=opt.hk,
                   autosave=opt.au,
                   border=opt.bo,
                   rootcrop=opt.rc
                   )


def saveIcon():
    """Save an image as icon.

    This will show the Save Image dialog, with the multisave mode off and
    asking for an icon file name. Then save the current rendering to that file.
    """
    ## We should create a specialized input dialog, asking also for the size
    fn = draw.askNewFilename(filter=utils.fileDescription('icon'))
    if fn:
        image.saveIcon(fn,size=32)


def startMultiSave():
    """Start/change multisave mode."""
    saveImage(True)


def stopMultiSave():
    """Stop multisave mode."""
    image.save()

from imageViewer import ImageViewer
viewer = None
def showImage():
    """Display an image file."""
    global viewer
    fn = draw.askFilename(filter=utils.fileDescription('img'))
    if fn:
        viewer = ImageViewer(pf.app,fn)
        viewer.show()


def listAll():
    print(pf.PF)



def createMovieInteractive():
    """Create a movie from a saved sequence of images.

    """
    if not image.multisave:
        pf.warning('You need to start multisave mode first!')
        return

    names = image.multisave[0]
    glob = names.glob()

    res = draw.askItems(
        [ _I('files',glob),
          _I('encoder',choices=['mencoder','convert','ffmpeg']),
          _G('Mencoder',[
              _I('fps',10),
              _I('vbirate',800),
              ]),
          _G('Convert',[
              _I('delay',1),
              _I('colors',256),
              ]),
          ],
        enablers = [
            ('encoder','mencoder','Mencoder'),
            ('encoder','convert','Convert'),
          ])
    if not res:
        return

    pf.GUI.setBusy()
    image.createMovie(**res)
    pf.GUI.setBusy(False)


_recording_pid = 0

def recordSession(stop=0):
    """Record the current pyFormex session."""
    global _recording_pid
    from guimain import hasDRI

    print("RECORDING with dri=%s" % pf.options.dri)

    ok = utils.checkExternal('recordmydesktop')
    if not ok:
        return
    if hasDRI():
        if not draw.ack("Recording the session while using DRI may fail to correctly capture the OpenGL canvas. We recommend starting pyformex with the --nodri option to do session recording (at the expense of a slower execution). If you click YES I will nevertheless go ahead with recording."):
            return

    fn = draw.askFilename(pf.cfg['workdir'],"Theora video files (*.ogv)",exist=False)
    if not fn:
        return

    print("Recording your session to file %s" % fn)
    x,y,w,h = pf.GUI.XGeometry()
    cmd = "recordmydesktop -x %s -y %s --width %s --height %s --no-frame -o %s" % (x,y,w,h,fn)
    print(cmd)
    pid = utils.spawn(cmd)
    print("Recording pid = %s" % pid)
    _recording_pid = pid


def stopRecording(stop=15):
    global _recording_pid
    print("STOP RECORDING")
    if _recording_pid:
        # Was recording: finish it
        utils.killProcesses([_recording_pid],stop)
        # should check that it has stopped
        _recording_pid = 0


def abortRecording():
    stopRecording(6)



MenuData = [
    ## (_('&Open project'),openProject),
    (_('&Start new project'),createProject),
    (_('&Open existing project'),openExistingProject),
    (_('&Import a project'),importProject),
    (_('&Set current script as AutoScript'),setAutoScript),
    (_('&Remove the AutoScript'),removeAutoScript),
    (_('&Set current script as AutoFile'),setAutoFile),
    (_('&Remove the AutoFile'),removeAutoFile),
    (_('&List project contents'),listAll),
    (_('&Save project'),saveProject),
    (_('&Save project As'),saveAsProject),
    ## (_('&Close project without saving'),closeProjectWithoutSaving),
    (_('&Clear project'),clearProject),
    (_('&Close project'),closeProject),
    ('---',None),
    (_('&Convert Project File'),convertProjectFile),
    (_('&Uncompress Project File'),uncompressProjectFile),
    ('---',None),
    (_('&Change workdir'),draw.askDirname),
    (_('&Create new script'),createScript),
    (_('&Open existing script'),openScript),
    (_('&Edit current script/app'),editApp),
    (_('&Run current script/app'),draw.play),
    (_('&Reload and run current app'),draw.replay),
    ## (_('&Unload all but current app'),),
    (_('---1'),None),
    (_('&Save Image'),saveImage),
    (_('Start &MultiSave'),startMultiSave),
    (_('Save &Next Image'),image.saveNext),
    (_('Create &Movie'),createMovieInteractive),
    (_('&Stop MultiSave'),stopMultiSave),
    (_('&Save as Icon'),saveIcon),
    (_('&Show Image'),showImage),
    (_('&Record Session'),[
        (_('&Start Recording'),recordSession),
        (_('&Stop Recording'),stopRecording),
        (_('&Abort Recording'),abortRecording),
        ]),
    (_('---2'),None),
    (_('E&xit'),draw.closeGui),
]

#onExit(closeProject)

# End
