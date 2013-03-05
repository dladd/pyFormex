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
"""Basic pyFormex script functions

The :mod:`script` module provides the basic functions available
in all pyFormex scripts. These functions are available in GUI and NONGUI
applications, without the need to explicitely importing the :mod:`script`
module.
"""
from __future__ import print_function

import pyformex as pf
import formex
import geomfile
import utils
from project import Project
from geometry import Geometry

########################
# Imported here only to make available in scripts
from olist import List
from mesh import Mesh
from plugins.trisurface import TriSurface

########################

import threading,os,copy,re,time

######################### Exceptions #########################################

class _Exit(Exception):
    """Exception raised to exit from a running script."""
    pass

class _ExitAll(Exception):
    """Exception raised to exit pyFormex from a script."""
    pass

class _ExitSeq(Exception):
    """Exception raised to exit from a sequence of scripts."""
    pass

class _TimeOut(Exception):
    """Exception raised to timeout from a dialog widget."""
    pass


############################# Globals for scripts ############################


def Globals():
    """Return the globals that are passed to the scripts on execution.

    When running pyformex with the --nogui option, this contains all the
    globals defined in the module formex (which include those from
    coords, arraytools and numpy.

    When running with the GUI, this also includes the globals from gui.draw
    (including those from gui.color).

    Furthermore, the global variable __name__ will be set to either 'draw'
    or 'script' depending on whether the script was executed with the GUI
    or not.
    """
    # :DEV it is not a good idea to put the pf.PF in the globals(),
    # because pf.PF may contain keys that are not acceptible as
    # Python names
    g = {}
    g.update(globals())
    if pf.GUI:
        from gui import draw
        g.update(draw.__dict__)
    g.update(formex.__dict__)
    return g


def export(dic):
    """Export the variables in the given dictionary."""
    pf.PF.update(dic)


def export2(names,values):
    """Export a list of names and values."""
    export(dict(zip(names,values)))


def forget(names):
    """Remove the global variables specified in list."""
    g = pf.PF
    for name in names:
        if name in g:
            del g[name]


def forgetAll():
    """Delete all the global variables."""
    pf.PF = {}


def rename(oldnames,newnames):
    """Rename the global variables in oldnames to newnames."""
    g = pf.PF
    for oldname,newname in zip(oldnames,newnames):
        if oldname in g:
            g[newname] = g[oldname]
            del g[oldname]


def listAll(clas=None,like=None,filtr=None,dic=None,sort=False):
    """Return a list of all objects in dictionay that match criteria.

    - dic: a dictionary object, defaults to pyformex.PF
    - clas: a class name: if specified, only instances of this class will be
      returned
    - like: a string: if given, only object names starting with this string
      will be returned
    - filtr: a function taking an object name as parameter and returning True
      or False. If specified, only objects passing the test will be returned.

    The return value is a list of keys from dic.
    """
    if dic is None:
        dic = pf.PF

    names = dic.keys()
    if clas is not None:
        names = [ n for n in names if isinstance(dic[n],clas) ]
    if like is not None:
        names = [ n for n in names if n.startswith(like) ]
    if filtr is not None:
        names = [ n for n in names if filtr(n) ]
    if sort:
        names.sort()
    return names


def named(name):
    """Returns the global object named name."""
    if name in pf.PF:
        dic = pf.PF
    else:
        raise NameError,"Name %s is not in pyformex.PF" % name
    return dic[name]


def getcfg(name):
    """Return a value from the configuration."""
    return pf.cfg.get(name,None)


#################### Interacting with the user ###############################

def ask(question,choices=None,default=''):
    """Ask a question and present possible answers.

    If no choices are presented, anything will be accepted.
    Else, the question is repeated until one of the choices is selected.
    If a default is given and the value entered is empty, the default is
    substituted.
    Case is not significant, but choices are presented unchanged.
    If no choices are presented, the string typed by the user is returned.
    Else the return value is the lowest matching index of the users answer
    in the choices list. Thus, ask('Do you agree',['Y','n']) will return
    0 on either 'y' or 'Y' and 1 on either 'n' or 'N'.
    """
    if choices:
        question += " (%s) " % ', '.join(choices)
        choices = [ c.lower() for c in choices ]
    while 1:
        res = raw_input(question)
        if res == '' and default:
            res = default
        if not choices:
            return res
        try:
            return choices.index(res.lower())
        except ValueError:
            pass

def ack(question):
    """Show a Yes/No question and return True/False depending on answer."""
    return ask(question,['Y','N']) == 0


def error(message):
    """Show an error message and wait for user acknowlegement."""
    print("pyFormex Error: "+message)
    if not ack("Do you want to continue?"):
        exit()

def warning(message):
    print("pyFormex Warning: "+message)
    if not ack("Do you want to continue?"):
        exit()

def showInfo(message):
    print("pyFormex Info: "+message)

message = pf.message

def system(cmdline,result='output'):
    """Run a command and return its output.

    If result == 'status', the exit status of the command is returned.
    If result == 'output', the output of the command is returned.
    If result == 'both', a tuple of status and output is returned.
    """
    sta,out,err = utils.system(cmdline)
    if result == 'status':
        return sta
    elif result == 'output':
        return out
    elif result == 'both':
        return sta,out


########################### PLAYING SCRIPTS ##############################

scriptThread = None
exitrequested = False
starttime = 0.0

scriptInit = None # can be set to execute something before each script

# BV: do we need this??
#pye = False

def scriptLock(id):
    pf.debug("Setting script lock %s" %id,pf.DEBUG.SCRIPT)
    pf.scriptlock |= set([id])
    #print(pf.scriptlock)

def scriptRelease(id):
    pf.debug("Releasing script lock %s" %id,pf.DEBUG.SCRIPT)
    pf.scriptlock -= set([id])
    #print(pf.scriptlock)


def playScript(scr,name=None,filename=None,argv=[],pye=False):
    """Play a pyformex script scr. scr should be a valid Python text.

    There is a lock to prevent multiple scripts from being executed at the
    same time. This implies that pyFormex scripts can currently not be
    recurrent.
    If a name is specified, set the global variable pyformex.scriptName to it
    when the script is started.
    If a filename is specified, set the global variable __file__ to it.
    """
    utils.warn('print_function')
    global exportNames,starttime
    global exitrequested
    # (We only allow one script executing at a time!)
    # and scripts are non-reentrant
    global scriptThread
    if scriptThread is not None and scriptThread.isAlive():
        pf.message("Not executing this script because another one is already running")
        return


    if len(pf.scriptlock) > 0:
        pf.message("!!Not executing because a script lock has been set: %s" % pf.scriptlock)
        #print(pf.scriptlock)
        return

    scriptLock('__auto__')
    exitrequested = False

    if pf.GUI:
        pf.GUI.startRun()

    # Get the globals
    g = Globals()
    if pf.GUI:
        modname = 'draw'
    else:
        modname = 'script'
    g.update({'__name__':modname})
    if filename:
        g.update({'__file__':filename})
    g.update({'argv':argv})

    # Now we can execute the script using these collected globals
    exportNames = []
    pf.scriptName = name
    exitall = False

    #starttime = time.clock()
    try:
        try:
            if pye:
                if type(scr) is file:
                     scr = scr.read() + '\n'
                n = (len(scr)+1) // 2
                scr = utils.mergeme(scr[:n],scr[n:])
            exec scr in g

        except _Exit:
            #print "EXIT FROM SCRIPT"
            pass
        except _ExitAll:
            exitall = True
        except:
            raise

    finally:
        # honour the exit function
        if 'atExit' in g:
            atExit = g['atExit']
            try:
                atExit()
            except:
                pf.debug('Error while calling script exit function',pf.DEBUG.SCRIPT)

        if pf.cfg['autoglobals']:
            exportNames.extend(listAll(clas=Geometry,dic=g))
        pf.PF.update([(k,g[k]) for k in exportNames])

        scriptRelease('__auto__') # release the lock
        if pf.GUI:
            pf.GUI.stopRun()

    if exitall:
        pf.debug("Calling quit() from playscript",pf.DEBUG.SCRIPT)
        quit()


def force_finish():
    pf.scriptlock = set() # release all script locks (in case of an error)
    #print(pf.scriptlock)


def breakpt(msg=None):
    """Set a breakpoint where the script can be halted on a signal.

    If an argument is specified, it will be written to the message board.

    The exitrequested signal is usually emitted by pressing a button in the GUI.
    """
    global exitrequested
    if exitrequested:
        if msg is not None:
            pf.message(msg)
        exitrequested = False # reset for next time
        raise _Exit


def raiseExit():
    pf.debug("RAISING _Exit",pf.DEBUG.SCRIPT)
    if pf.GUI:
        pf.GUI.drawlock.release()
    raise _Exit,"EXIT REQUESTED FROM SCRIPT"


def enableBreak(mode=True):
    if pf.GUI:
        pf.GUI.enableButtons(pf.GUI.actions,['Stop'],mode)


def stopatbreakpt():
    """Set the exitrequested flag."""
    global exitrequested
    exitrequested = True


def convertPrintSyntax(filename):
    """Convert a script to using the print function"""
    sta,out = utils.runCommand("2to3 -f print -wn %s" % filename)
    if sta:
        # Conversion error: show what is going on
        print(out)
    return sta == 0


def checkPrintSyntax(filename):
    """Check whether the script in the given files uses print function syntax.

    Returns the compiled object if no error was found during compiling.
    Returns the filename if an error was found and correction has been
    attempted.
    Raises an exception if an error is found and no correction attempted.
    """
    with open(filename,'r') as f:
        try:
            script = f.read()
            scr = compile(script,filename,'exec')
            return scr
        except SyntaxError as err:
            if re.compile('.*print +[^ (]').match(err.text):
                ans = pf.warning("""..

Syntax error in line %s of %s::

  %s

It looks like your are using a print statement instead of the print function.
In order to prepare you for the future (Python3), pyFormex already enforces
the use of the print function.
This means that you have to change any print statement from::

    print something

into a print function call::

    print(something)

You can try an automatic conversion with the command::

    2to3 -f print -%s

If you want, I can run this command for you now. Beware though!
This will overwrite the contents of file %s.

Also, the automatic conversion is not guaranteed to work, because
there may be other errors.
""" % (err.lineno,err.filename,err.text,filename,filename),actions=['Not this time','Convert now'],)
                if ans == 'Convert now':
                    print(ans)
                    if convertPrintSyntax(filename):
                        message("Script properly converted, now running the converted script")
                        return filename
                raise


def runScript(fn,argv=[]):
    """Play a formex script from file fn.

    fn is the name of a file holding a pyFormex script.
    A list of arguments can be passed. They will be available under the name
    argv. This variable can be changed by the script and the resulting argv
    is returned to the caller.
    """
    from timer import Timer
    t = Timer()
    msg = "Running script (%s)" % fn
    if pf.GUI:
        pf.GUI.scripthistory.add(fn)
        pf.board.write(msg,color='red')
    else:
        message(msg)
    pf.debug("  Executing with arguments: %s" % argv,pf.DEBUG.SCRIPT)
    pye = fn.endswith('.pye')
    if pf.GUI and getcfg('check_print'):
        pf.debug("Testing script for use of print function",pf.DEBUG.SCRIPT)
        scr = checkPrintSyntax(fn)
        #
        # TODO: if scr is a compiled object, we could just execute it
        #

    res = playScript(file(fn,'r'),fn,fn,argv,pye)
    pf.debug("  Arguments left after execution: %s" % argv,pf.DEBUG.SCRIPT)
    msg = "Finished script %s in %s seconds" % (fn,t.seconds())
    if pf.GUI:
        pf.board.write(msg,color='red')
    else:
        message(msg)
    return res


def runApp(appname,argv=[],refresh=False):
    global exitrequested
    if len(pf.scriptlock) > 0:
        pf.message("!!Not executing because a script lock has been set: %s" % pf.scriptlock)
        #print(pf.scriptlock)
        return

    import apps
    from timer import Timer
    t = Timer()
    pf.message("Loading application %s with refresh=%s" % (appname,refresh))
    app = apps.load(appname,refresh=refresh)
    if app is None:
        errmsg = "An  error occurred while loading application %s" % appname
        if pf.GUI:
            if apps._traceback and pf.cfg['showapploaderrors']:
                print(apps._traceback)

            from gui import draw
            fn = apps.findAppSource(appname)
            if os.path.exists(fn):
                errmsg += "\n\nYou may try executing the application as a script,\n  or you can load the source file in the editor."
                res = draw.ask(errmsg,choices=['Run as script', 'Load in editor', "Don't bother"])
                if res[0] in 'RL':
                    if res[0] == 'L':
                        draw.editFile(fn)
                    elif res[0] == 'R':
                        pf.GUI.setcurfile(fn)
                        draw.runScript(fn)
            else:
                errmsg += "and I can not find the application source file."
                draw.error(errmsg)
        else:
            error(errmsg)

        return

    if hasattr(app,'_status') and app._status == 'unchecked':
        pf.warning("This looks like an Example script that has been automatically converted to the pyFormex Application model, but has not been checked yet as to whether it is working correctly in App mode.\nYou can help here by running and rerunning the example, checking that it works correctly, and where needed fixing it (or reporting the failure to us). If the example runs well, you can change its status to 'checked'")

    scriptLock('__auto__')
    msg = "Running application '%s' from %s" % (appname,app.__file__)
    pf.scriptName = appname
    if pf.GUI:
        pf.GUI.startRun()
        pf.GUI.apphistory.add(appname)
        pf.board.write(msg,color='green')
    else:
        message(msg)
    pf.debug("  Passing arguments: %s" % argv,pf.DEBUG.SCRIPT)
    app._args_ = argv
    try:
        try:
            res = app.run()
        except _Exit:
            pass
        except _ExitSeq:
            exitrequested = True
        except:
            raise
    finally:
        if hasattr(app,'atExit'):
            app.atExit()
        if pf.cfg['autoglobals']:
            g = app.__dict__
            exportNames = listAll(clas=Geometry,dic=g)
            pf.PF.update([(k,g[k]) for k in exportNames])
        scriptRelease('__auto__') # release the lock
        if pf.GUI:
            pf.GUI.stopRun()

    pf.debug("  Arguments left after execution: %s" % argv,pf.DEBUG.SCRIPT)
    msg = "Finished %s in %s seconds" % (appname,t.seconds())
    if pf.GUI:
        pf.board.write(msg,color='green')
    else:
        message(msg)
    pf.debug("Memory: %s" % vmSize(),pf.DEBUG.MEM)


def runAny(appname=None,argv=[],step=False,refresh=False):
    """Run the current pyFormex application or script file.

    This function does nothing if no appname/filename is passed or no current
    script/app was set.
    If arguments are given, they are passed to the script. If `step` is True,
    the script is executed in step mode. The 'refresh' parameter will reload
    the app.
    """
    if appname is None:
        appname = pf.cfg['curfile']
    if not appname:
        return

    #print "RUNNING %s" % appname
    if scriptInit:
        #print "INITFUNC"
        scriptInit()

    if pf.GUI:
        pf.GUI.setcurfile(appname)

    if utils.is_script(appname):
        #print "RUNNING SCRIPT %s" % appname
        return runScript(appname,argv)
    else:
        #print "RUNNING APP %s" % appname
        return runApp(appname,argv,refresh)


## def runAll(applist,refresh=False):
##     """Run all the scripts/apps in given list."""
##     pf.GUI.enableButtons(pf.GUI.actions,['Stop'],True)
##     for f in applist:
##         while pf.scriptlock:
##             #print(pf.scriptlock)
##             print("WAITING BECAUSE OF SCRIPT LOCK")
##             time.sleep(5)
##         runAny(f,refresh=refresh)
##         if exitrequested:
##             break
##     pf.GUI.enableButtons(pf.GUI.actions,['Stop'],False)


def exit(all=False):
    """Exit from the current script or from pyformex if no script running."""
    #print("DRAW.EXIT")
    #print(pf.scriptlock)
    if len(pf.scriptlock) > 0:
        if all:
            raise _ExitAll # ask exit from pyformex
        else:
            #print "RAISE _EXIT"
            raise _Exit # ask exit from script only


def quit():
    """Quit the pyFormex program

    This is a hard exit from pyFormex. It is normally not called
    directly, but results from an exit(True) call.
    """
    if pf.app and pf.app_started: # quit the QT app
        pf.debug("draw.exit called while no script running",pf.DEBUG.SCRIPT)
        pf.app.quit() # closes the GUI and exits pyformex
    else: # the QT app didn't even start
        sys.exit(0) # use Python to exit pyformex


def processArgs(args):
    """Run the application without gui.

    Arguments are interpreted as names of script files, possibly interspersed
    with arguments for the scripts.
    Each running script should pop the required arguments from the list.
    """
    res = 0
    while len(args) > 0:
        fn = args.pop(0)
        if fn.endswith('.pye'):
            pass
        elif not os.path.exists(fn) or not utils.is_pyFormex(fn):
            pf.message("Skipping %s: does not exist or is not a pyFormex script" % fn)
            continue
        res = runScript(fn,args)
        if res and pf.GUI:
            pf.message("Error during execution of script %s" % fn)

    return res


def setPrefs(res,save=False):
    """Update the current settings (store) with the values in res.

    res is a dictionary with configuration values.
    The current settings will be update with the values in res.

    If save is True, the changes will be stored to the user's
    configuration file.
    """
    pf.debug("Accepted settings:\n%s"%res,pf.DEBUG.CONFIG)
    for k in res:
        pf.cfg[k] = res[k]
        if save and pf.prefcfg[k] != pf.cfg[k]:
            pf.prefcfg[k] = pf.cfg[k]

    pf.debug("New settings:\n%s"%pf.cfg,pf.DEBUG.CONFIG)
    if save:
        pf.debug("New preferences:\n%s"%pf.prefcfg,pf.DEBUG.CONFIG)


########################## print information ################################


def printall():
    """Print all Formices in globals()"""
    print("Formices currently in globals():\n%s" % listAll(clas=formex.Formex))


def printglobals():
    print(Globals())


def printglobalnames():
    a = Globals().keys()
    a.sort()
    print(a)


def printconfig():
    print("Reference Configuration: " + str(pf.refcfg))
    print("Preference Configuration: " + str(pf.prefcfg))
    print("User Configuration: " + str(pf.cfg))


def printdetected():
    print(utils.reportDetected())


def printLoadedApps():
    import apps,sys
    loaded = apps.listLoaded()
    refcnt = [ sys.getrefcount(sys.modules[k]) for k in loaded ]
    print(', '.join([ "%s (%s)" % (k,r) for k,r in zip(loaded,refcnt)]))


def vmSize():
    import os
    return os.popen('ps h o vsize %s'%os.getpid()).read().strip()


def printVMem(msg='MEMORY'):
    print('%s: VmSize=%skB'%(msg,vmSize()))


### Utilities

def isWritable(path):
    """Check that the specified path is writable.

    BEWARE: this only works if the path exists!
    """
    return os.access(path,os.W_OK)


def chdir(path,create=False):
    """Change the current working directory.

    If path exists and it is a directory name, make it the current directory.
    If path exists and it is a file name, make the containing directory the
    current directory.
    If path does not exist and create is True, create the path and make it the
    current directory. If create is False, raise an Error.

    Parameters:

    - `path`: pathname of the directory or file. If it is a file, the name of
      the directory holding the file is used. The path can be an absolute
      or a relative pathname. A '~' character at the start of the pathname will
      be expanded to the user's home directory.

    - `create`: bool. If True and the specified path does not exist, it will
      be created. The default is to do nothing if the specified path does
      not exist.

    The changed to current directory is stored in the user's preferences
    for persistence between pyFormex invocations.

    """
    path = utils.tildeExpand(path)
    if os.path.exists(path):
        if not os.path.isdir(path):
            path = os.path.dirname(os.path.abspath(path))
    else:
        if not create or not mkdir(path):
            raise ValueError,"The path %s does not exist" % path
    os.chdir(path)
    setPrefs({'workdir':path},save=True)
    if pf.GUI:
        pf.GUI.setcurdir()
    pwdir()


def pwdir():
    """Print the current working directory.

    """
    pf.message("Current workdir is %s" % os.getcwd())


def mkdir(path):
    """Create a new directory.

    Create a new directory, including any needed parent directories.

    - `path`: pathname of the directory to create, either an absolute
      or relative path. A '~' character at the start of the pathname will
      be expanded to the user's home directory. If the path exists, the
      function returns True without doing anything.

    Returns True if the pathname exists (before or after).
    """
    path = utils.tildeExpand(path)
    if not path or os.path.exists(path) and os.path.isdir(path):
        return True
    if os.path.exists(path):
        raise ValueError,"The path %s does exists already" % path
    mkdir(os.path.dirname(path))
    os.mkdir(path)
    return os.path.exists(path)


def mkpdir(path):
    """Make sure the parent directory of path exists.

    """
    return mkdir(os.path.dirname(path))



def runtime():
    """Return the time elapsed since start of execution of the script."""
    return time.clock() - starttime


def startGui(args=[]):
    """Start the gui"""
    if pf.GUI is None:
        pf.debug("Starting the pyFormex GUI",pf.DEBUG.GUI)
        from gui import guimain
        if guimain.startGUI(args) == 0:
            guimain.runGUI()


def checkRevision(rev,comp='>='):
    """Check the pyFormex revision number.

    - rev: a positive integer.
    - comp: a string specifying a comparison operator.

    By default, this function returns True if the pyFormex revision
    number is equal or larger than the specified number.

    The comp argument may specify another comparison operator.

    If pyFormex is unable to find its revision number (this is the
    case on very old versions) the test returns False.
    """
    try:
        cur = int(utils.splitStartDigits(pf.__revision__)[0])
        return eval("%s %s %s" % (cur,comp,rev))
    except:
        return False


def requireRevision(rev,comp='>='):
    """Require a specified pyFormex revision number.

    The arguments are like checkRevision. Ho9wever, this function will
    raise an error if the requirement fails.
    """
    if not checkRevision(rev,comp):
        raise RuntimeError,"Your current pyFormex revision (%s) does not pass the test %s %s" % (pf.__revision__,comp,rev)



################### read and write files #################################

def writeGeomFile(filename,objects,sep=' ',mode='w',shortlines=False):
    """Save geometric objects to a pyFormex Geometry File.

    A pyFormex Geometry File can store multiple geometrical objects in a
    native format that can be efficiently read back into pyformex.
    The format is portable over different pyFormex versions and
    even to other software.

    - `filename`: the name of the file to be written. If it ends with '.gz'
      the file will be compressed with gzip. If a file with the given name
      minus the trailing '.gz' exists, it will be destroyed.
    - `objects`: a list or a dictionary. If it is a dictionary,
      the objects will be saved with the key values as there names.
      Objects that can not be exported to a Geometry File will be
      silently ignored.
    - `mode`: can be set to 'a' to append to an existing file.
    - `sep`: the string used to separate data. If set to an empty
      string, the data will be written in binary format and the resulting file
      will be smaller but less portable.

    Returns the number of objects written to the file.
    """
    gzip = filename.endswith('.gz')
    if gzip:
        filename = filename[:-3]
    f = geomfile.GeometryFile(filename,mode='w',sep=sep)
    if shortlines:
        f.fmt = {'i':'%i ','f':'%f '}
    f.write(objects)
    f.close()
    if gzip:
        utils.gzip(filename)
    return len(objects)


def readGeomFile(filename):
    """Read a pyFormex Geometry File.

    A pyFormex Geometry File can store multiple geometrical objects in a
    native format that can be efficiently read back into pyformex.
    The format is portable over different pyFormex versions and
    even to other software.

    - `filename`: the name of an existing pyFormex Geometry File. If the
      filename ends on '.gz', it is considered to be a gzipped file and
      will be uncompressed transparently during the reading.

    Returns a dictionary with the geometric objects read from the file.
    If object names were stored in the file, they will be used as the keys.
    Else, default names will be provided.
    """
    gzip = filename.endswith('.gz')
    if gzip:
        filename = utils.gunzip(filename,unzipped='',remove=False)
    f = geomfile.GeometryFile(filename,mode='r')
    objects = f.read()
    if gzip:
        utils.removeFile(filename)
    return objects


#### End
