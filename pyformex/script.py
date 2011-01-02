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
"""Basic pyFormex script functions

The :mod:`script` module provides the basic functions available
in all pyFormex scripts. These functions are available in GUI and NONGUI
applications, without the need to explicitely importing the :mod:`script`
module.
"""

import pyformex as pf
import formex
import geomfile
import utils

########################
# Imported here only to make it available in scripts
from mesh import Mesh

########################

import threading,os,commands,copy,re,time

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

    This basically contains the globals defined in draw.py, colors.py,
    and formex.py, as well as the globals from numpy.
    It also contains the definitions put into the pyformex.PF, by
    preference using the export() function.
    During execution of the script, the global variable __name__ will be
    set to either 'draw' or 'script' depending on whether the script
    was executed in the 'draw' module (--gui option) or the 'script'
    module (--nogui option).
    """
    g = copy.copy(pf.PF)
    g.update(globals())
    if pf.GUI:
        from gui import colors,draw
        g.update(colors.__dict__)
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
        if g.has_key(name):
            del g[name]


def forgetAll():
    """Delete all the global variables."""
    pf.PF = {}
    

def rename(oldnames,newnames):
    """Rename the global variables in oldnames to newnames."""
    g = pf.PF
    for oldname,newname in zip(oldnames,newnames):
        if g.has_key(oldname):
            g[newname] = g[oldname]
            del g[oldname]


def listAll(clas=None,like=None,filter=None,dic=None):
    """Return a list of all objects in dictionay that match criteria.

    - dic: a dictionary object, defaults to pyformex.PF
    - clas: a class name: if specified, only instances of this class will be
      returned
    - like: a string: if given, only object names starting with this string
      will be returned
    - filter: a function taking an object name as parameter and returning True
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
    if filter is not None:
        names = [ n for n in names if filter(n) ]
    return names


def named(name):
    """Returns the global object named name."""
    if pf.PF.has_key(name):
        dic = pf.PF
    elif pf._PF_.has_key(name):
        pf.debug("Found %s in pyformex._PF_" % name)
        dic = pf._PF_
    else:
        raise NameError,"Name %s is in neither pyformex.PF nor pyformex._PF_" % name
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

##def log(s):
##    """Display a message in the terminal."""
##    print(s)

# message is the preferred function to send text info to the user.
# The default message handler is set here.
# Best candidates are log/info
message = pf.message

def system(cmdline,result='output'):
    """Run a command and return its output.

    If result == 'status', the exit status of the command is returned.
    If result == 'output', the output of the command is returned.
    If result == 'both', a tuple of status and output is returned.
    """
    if result == 'status':
        return os.system(cmdline)
    elif result == 'output':
        return commands.getoutput(cmdline)
    elif result == 'both':
        return commands.getstatusoutput(cmdline)


########################### PLAYING SCRIPTS ##############################

sleep = time.sleep

scriptDisabled = False
scriptRunning = False
scriptThread = None
exitrequested = False
stepmode = False
starttime = 0.0
pye = False


def executeScript(scr,glob):
    """Execute a Python script in specified globals."""
    ## print "SCRIPT"
    ## print scr
    ## print "GLOBALS"
    ## print glob
    exec scr in glob
 
def playScript(scr,name=None,filename=None,argv=[],pye=False):
    """Play a pyformex script scr. scr should be a valid Python text.

    There is a lock to prevent multiple scripts from being executed at the
    same time. This implies that pyFormex scripts can currently not be
    recurrent.
    If a name is specified, set the global variable pyformex.scriptName to it
    when the script is started.
    If a filename is specified, set the global variable __file__ to it.
    
    If step==True, an indefinite pause will be started after each line of
    the script that starts with 'draw'. Also (in this case), each line
    (including comments) is echoed to the message board.
    """
    from geometry import Geometry
    global scriptDisabled,scriptRunning,exitrequested
    #pf.debug('SCRIPT MODE %s,%s,%s'% (scriptDisabled,scriptRunning,exitrequested))
    # (We only allow one script executing at a time!)
    # and scripts are non-reentrant
    global scriptThread
    if scriptThread is not None and scriptThread.isAlive():
        pf.message("Not executing this script because another one is already running")
        return
        
    if scriptRunning or scriptDisabled :
        pf.message("Not executing this script because another one is already running")
        return
    
    scriptRunning = True
    exitrequested = False

    if pf.GUI:
        global stepmode,exportNames,starttime
        #pf.debug('GUI SCRIPT MODE %s'% (stepmode))
        pf.GUI.drawlock.allow()
        pf.canvas.update()
        pf.GUI.actions['Play'].setEnabled(False)
        pf.GUI.actions['Continue'].setEnabled(True)
        pf.GUI.actions['Stop'].setEnabled(True)
        pf.app.processEvents()
    
    # Get the globals
    g = Globals()
    if pf.GUI:
        modname = 'draw'
        # by default, we run the script in the current GUI viewport
        pf.canvas = pf.GUI.viewports.current
    else:
        modname = 'script'
    g.update({'__name__':modname})
    if filename:
        g.update({'__file__':filename})
    g.update({'argv':argv})

    # Make this directory available
    pf._PF_ = g

    # Now we can execute the script using these collected globals
    exportNames = []
    pf.scriptName = name
    exitall = False

    starttime = time.clock()
    pf.debug('STARTING SCRIPT (%s)' % starttime)
    #pf.debug(scr)
    #pf.debug(pye)
    try:
        try:
            if pf.GUI and stepmode:
                #pf.debug("STEPPING THROUGH SCRIPT")
                step_script(scr,g,True)
            else:
                if pye:
                    if type(scr) is file:
                         scr = scr.read() + '\n'
                    n = len(scr) // 2
                    scr = utils.mergeme(scr[:n],scr[n:])
                if pf.options.executor:
                    scriptThread = threading.Thread(None,executeScript,'script-0',(scr,g))
                    scriptThread.daemon = True
                    print "OK, STARTING THREAD"
                    scriptThread.start()
                    print "OK, STARTED THREAD"
                else:
                    exec scr in g

        except _Exit:
            pass
        except _ExitAll:
            exitall = True
        except:
            raise
            
    finally:
        # honour the exit function
        if g.has_key('atExit'):
            atExit = g['atExit']
            try:
                atExit()
            except:
                pf.debug('Error while calling script exit function')
                
        if pf.cfg['autoglobals']:
            exportNames.extend(listAll(clas=Geometry,dic=g))
        pf.PF.update([(k,g[k]) for k in exportNames])

        scriptRunning = False # release the lock in case of an error
        elapsed = time.clock() - starttime
        pf.debug('SCRIPT RUNTIME : %s seconds' % elapsed)
        if pf.GUI:
            stepmode = False
            pf.GUI.drawlock.release() # release the lock
            pf.GUI.actions['Play'].setEnabled(True)
            #pf.GUI.actions['Step'].setEnabled(False)
            pf.GUI.actions['Continue'].setEnabled(False)
            pf.GUI.actions['Stop'].setEnabled(False)

    if exitall:
        pf.debug("Calling exit() from playscript")
        exit()


def force_finish():
    global scriptRunning,stepmode
    # print "FORCE FINISH"
    scriptRunning = False # release the lock in case of an error
    stepmode = False


def step_script(s,glob,paus=True):
    buf = ''
    for line in s:
        if buf.endswith('\\'):
            buf[-1:] = line
            break
        else:
            buf += line
        if paus and (line.strip().startswith('draw') or
                     line.find('draw(') >= 0  or
                     line.strip().startswith('view') or
                     line.find('view(') >= 0 ):
            pf.GUI.drawlock.block()
            message(buf)
            exec(buf) in glob
            buf = ''

    showInfo("Finished stepping through script!")


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


## def raiseExit():
##     pf.debug("RAISED EXIT")
##     raise _Exit


def enableBreak(mode=True):
    pf.GUI.actions['Stop'].setEnabled(mode)


def stopatbreakpt():
    """Set the exitrequested flag."""
    global exitrequested
    exitrequested = True


def playFile(fn,argv=[]):
    """Play a formex script from file fn.

    fn is the name of a file holding a pyFormex script.
    A list of arguments can be passed. They will be available under the name
    argv. This variable can be changed by the script and the resulting argv
    is returned to the caller.
    """
    from timer import Timer
    t = Timer()
    message("Running script (%s)" % fn)
    pf.debug("  Executing with arguments: %s" % argv)
    res = playScript(file(fn,'r'),fn,fn,argv,fn.endswith('.pye'))
    pf.debug("  Arguments left after execution: %s" % argv)
    message("Finished script %s in %s seconds" % (fn,t.seconds()))
    return res


def play(fn=None,argv=[],step=False):
    """Play the pyFormex script with filename `fn` or the current script file.

    This function does nothing if no filename is passed or no current
    scriptfile was set.
    If arguments are given, they are passed to the script. If `step` is True,
    the script is executed in step mode.
    """
    global stepmode
    if not fn:
        if pf.GUI.canPlay:
            fn = pf.cfg['curfile']
        else:
            return
    stepmode = step
    pf.GUI.history.add(fn)
    stepmode = step
    return playFile(fn,argv)


def exit(all=False):
    """Exit from the current script or from pyformex if no script running."""
    #print "SCRIPT.EXIT"
    if scriptRunning:
        if all:
            raise _ExitAll # exit from pyformex
        else:
            raise _Exit # exit from script only
    if pf.app and pf.app_started: # exit from GUI
        pf.debug("draw.exit called while no script running")
        pf.app.quit() # close GUI and exit pyformex
    else: # the gui didn't even start
        sys.exit(0) # exit from pyformex
        

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
        res = playFile(fn,args)
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
    pf.debug("Accepted settings:",res)
    for k in res:
        pf.cfg[k] = res[k]
        if save and pf.prefcfg[k] != pf.cfg[k]:
            pf.prefcfg[k] = pf.cfg[k]

    pf.debug("New settings:",pf.cfg)
    if save:
        pf.debug("New preferences:",pf.prefcfg)


########################## print information ################################
    

def printall():
    """Print all Formices in globals()"""
    print("Formices currently in globals():\n%s" % listAll(clas=formex.Formex))


def printglobals():
    print(globals())


def printglobalnames():
    a = globals().keys()
    a.sort()
    print(a)

    
def printconfig():
    print("Reference Configuration: " + str(pf.refcfg))
    print("Preference Configuration: " + str(pf.prefcfg))
    print("User Configuration: " + str(pf.cfg))


def printdetected():
    print(utils.reportDetected())

### Utilities

def writable(path):
    """Returns True if the specified path is writeable"""
    return os.access(path,os.W_OK)

def chdir(fn):
    """Change the current working directory.

    If fn is a directory name, the current directory is set to fn.
    If fn is a file name, the current directory is set to the directory
    holding fn.
    In either case, the current directory is stored in the user's preferences
    for persistence between pyFormex invocations.
    
    If fn does not exist, nothing is done.
    """
    if os.path.exists(fn):
        if not os.path.isdir(fn):
            fn = os.path.dirname(os.path.abspath(fn))
        os.chdir(fn)
        setPrefs({'workdir':fn},save=True)
        if pf.GUI:
            pf.GUI.setcurdir()
        pf.message("Your current workdir is %s" % os.getcwd())


def runtime():
    """Return the time elapsed since start of execution of the script."""
    return time.clock() - starttime


def startGui(args=[]):
    """Start the gui"""
    if pf.GUI is None:
        pf.debug("Starting the pyFormex GUI")
        from gui import guimain
        if guimain.startGUI(args) == 0:
            guimain.runGUI()


def isWritable(path):
    """Check that the specified path is writeable.

    BEWARE: this only works if the path exists!
    """
    return os.access(path,os.W_OK)


def checkRevision(rev,comp='>='):
    """Check that we have the requested revision number.

    Raises an error if the revision number of the running pyFormex does not
    pass the comparison test with the given revision number.

    rev: a positive integer.
    comp: a string used in the comparison.

    Default is to allow the specified revision and all later ones.
    """
    try:
        cur = int(utils.splitStartDigits(pf.__revision__.split()[1])[0])
        if not eval("%s %s %s" % (cur,comp,rev)):
            raise RuntimeError
    except:
        raise RuntimeError,"Your current pyFormex revision (%s) does not pass the test %s %s" % (pf.__revision__,comp,rev)
   
################### read and write files #################################

def writeGeomFile(filename,objects,sep=' ',mode='w'):
    """Save geometric objects to a pyFormex Geometry File.

    A pyFormex Geometry File can store multiple geometrical objects in a
    native format that can be efficiently read back into pyformex.
    The format is portable over different pyFormex versions and 
    even to other software.

    - `filename`: the name of the file to be written
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
    f = geomfile.GeometryFile(filename,mode='w',sep=sep)
    f.write(objects)
    f.close()
    return len(objects)
    

def readGeomFile(filename):
    """Read a pyFormex Geometry File.

    A pyFormex Geometry File can store multiple geometrical objects in a
    native format that can be efficiently read back into pyformex.
    The format is portable over different pyFormex versions and 
    even to other software.

    - `filename`: the name of an exisiting pyFormex Geometry File.
    
    Returns a dictionary with the geometric objects read from the file.
    If object names were stored in the file, they will be used as the keys.
    Else, default names will be provided.
    """
    f = geomfile.GeometryFile(filename,mode='r')
    return f.read()
    

#### End
