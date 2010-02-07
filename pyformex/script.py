#!/usr/bin/env python
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
"""Basic pyFormex script functions

The :mod:`script` module provides the basic functions available
in all pyFormex scripts. These functions are available in GUI and NONGUI
applications, without the need to explicitely importing the :mod:`script`
module.
"""

import pyformex
import formex
import geomfile
import utils

import threading,os,commands,copy,re,time

######################### Exceptions #########################################

class Exit(Exception):
    """Exception raised to exit from a running script."""
    pass    

class ExitAll(Exception):
    """Exception raised to exit pyFormex from a script."""
    pass    

class ExitSeq(Exception):
    """Exception raised to exit from a sequence of scripts."""
    pass    

class TimeOut(Exception):
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
    g = copy.copy(pyformex.PF)
    g.update(globals())
    if pyformex.GUI:
        from gui import colors,draw
        g.update(colors.__dict__)
        g.update(draw.__dict__)
    g.update(formex.__dict__)
    return g


def export(dic):
    """Export the variables in the given dictionary."""
    pyformex.PF.update(dic)


def export2(names,values):
    """Export a list of names and values."""
    export(dict(zip(names,values)))


def forget(names):
    """Remove the global variables specified in list."""
    g = pyformex.PF
    for name in names:
        if g.has_key(name):
            del g[name]
        

def rename(oldnames,newnames):
    """Rename the global variables in oldnames to newnames."""
    g = pyformex.PF
    for oldname,newname in zip(oldnames,newnames):
        if g.has_key(oldname):
            g[newname] = g[oldname]
            del g[oldname]


def listAll(clas=None,dic=None):
    """Return a list of all objects in dic that are of given clas.

    If no class is given, Formex objects are sought.
    If no dict is given, the objects from both pyformex.PF and locals()
    are returned.
    """
    if dic is None:
        dic = pyformex.PF

    if clas is None:
        return dic.keys()
    else:
        return [ k for k in dic.keys() if isinstance(dic[k],clas) ]


def named(name):
    """Returns the global object named name."""
    if pyformex.PF.has_key(name):
        dic = pyformex.PF
    elif pyformex._PF_.has_key(name):
        pyformex.debug("Found %s in pyformex._PF_" % name)
        dic = pyformex._PF_
    else:
        raise NameError,"Name %s is in neither pyformex.PF nor pyformex._PF_" % name
    return dic[name]


def getcfg(name):
    """Return a value from the configuration."""
    return pyformex.cfg.get(name,None)


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
message = pyformex.message

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
exitrequested = False
stepmode = False
starttime = 0.0
pye = False

 
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
    global scriptDisabled,scriptRunning,exitrcrequested
    #pyformex.debug('SCRIPT MODE %s,%s,%s'% (scriptDisabled,scriptRunning,exitrequested))
    # (We only allow one script executing at a time!)
    # and scripts are non-reentrant
    if scriptRunning or scriptDisabled :
        pyformex.message("Not executing this script because another one is already running")     
        return
    
    scriptRunning = True
    exitrequested = False

    if pyformex.GUI:
        global stepmode,exportNames,starttime
        #pyformex.debug('GUI SCRIPT MODE %s'% (stepmode))
        pyformex.GUI.drawlock.allow()
        pyformex.canvas.update()
        pyformex.GUI.actions['Play'].setEnabled(False)
        pyformex.GUI.actions['Continue'].setEnabled(True)
        pyformex.GUI.actions['Stop'].setEnabled(True)
        pyformex.app.processEvents()
    
    # Get the globals
    g = Globals()
    if pyformex.GUI:
        modname = 'draw'
    else:
        modname = 'script'
    g.update({'__name__':modname})
    if filename:
        g.update({'__file__':filename})
    g.update({'argv':argv})

    # Make this directory available
    pyformex._PF_ = g

    # Now we can execute the script using these collected globals
    exportNames = []
    pyformex.scriptName = name
    exitall = False

    starttime = time.clock()
    pyformex.debug('STARTING SCRIPT (%s)' % starttime)
    #pyformex.debug(scr)
    #pyformex.debug(pye)
    try:
        try:
            if pyformex.GUI and stepmode:
                #pyformex.debug("STEPPING THROUGH SCRIPT")
                step_script(scr,g,True)
            else:
                if pyformex.options.executor:
                    import sys
                    print(name,filename)
                    n = os.path.split(name)
                    m = os.path.basename(name)
                    m = os.path.basename(name)
                    print(n)
                    o = os.path.split(n[0])
                    print(o)
                    sys.path.insert(0,n[0])
                    print(sys.path)
                    print(m)
                    s = m.replace('.py','')
                    print(s)
                    __import__(s,g)
                else:
                    if pye:
                        if type(scr) is file:
                             scr = scr.read() + '\n'
                        n = len(scr) // 2
                        scr = utils.mergeme(scr[:n],scr[n:])
                    exec scr in g

            if pyformex.cfg['autoglobals']:
                exportNames.extend(listAll(clas=formex.Formex,dic=g))
            pyformex.PF.update([(k,g[k]) for k in exportNames])
        except Exit:
            pass
        except ExitAll:
            exitall = True
        except:
            raise
            
    finally:
        scriptRunning = False # release the lock in case of an error
        elapsed = time.clock() - starttime
        pyformex.debug('SCRIPT RUNTIME : %s seconds' % elapsed)
        if pyformex.GUI:
            stepmode = False
            pyformex.GUI.drawlock.release() # release the lock
            pyformex.GUI.actions['Play'].setEnabled(True)
            #pyformex.GUI.actions['Step'].setEnabled(False)
            pyformex.GUI.actions['Continue'].setEnabled(False)
            pyformex.GUI.actions['Stop'].setEnabled(False)

    if exitall:
        pyformex.debug("Calling exit() from playscript")
        exit()


def force_finish():
    global scriptRunning,stepmode
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
                     line.find('draw(') >= 0 ):
            pyformex.GUI.drawlock.block()
            message(buf)
            exec(buf) in glob
            buf = ''

    showInfo("Finished stepping through script!")


def breakpt(msg=None):
    """Set a breakpoint where the script can be halted on a signal.

    If an argument is specified, it will be written to the message board.

    The exitrequested signal is usually emitted by pressing a button in the GUI.
    In nongui mode the stopatbreakpt function can be called from another thread.
    """
    global exitrequested
    if exitrequested:
        if msg is not None:
            pyformex.message(msg)
        exitrequested = False # reset for next time
        raise Exit


def enableBreak(mode=True):
    pyformex.GUI.actions['Stop'].setEnabled(mode)


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
    message("Running script (%s)" % fn)
    pyformex.debug("  Executing with arguments: %s" % argv)
    res = playScript(file(fn,'r'),fn,fn,argv,fn.endswith('.pye'))
    pyformex.debug("  Arguments left after execution: %s" % argv)
    message("Finished script %s" % fn)
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
        if pyformex.GUI.canPlay:
            fn = pyformex.cfg['curfile']
        else:
            return
    stepmode = step
    pyformex.GUI.history.add(fn)
    stepmode = step
    return playFile(fn,argv)


def exit(all=False):
    """Exit from the current script or from pyformex if no script running."""
    if scriptRunning:
        if all:
            raise ExitAll # exit from pyformex
        else:
            raise Exit # exit from script only
    if pyformex.app and pyformex.app_started: # exit from GUI
        pyformex.debug("draw.exit called while no script running")
        pyformex.app.quit() # close GUI and exit pyformex
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
        elif not os.path.exists(fn) or not utils.isPyFormex(fn):
            pyformex.message("Skipping %s: does not exist or is not a pyFormex script" % fn)
            continue
        res = playFile(fn,args)
        if res and pyformex.GUI:
            pyformex.message("Error during execution of script %s" % fn)
        
    return res


########################## print information ################################
    
def formatInfo(F):
    """Return formatted information about a Formex."""
    bb = F.bbox()
    return """shape    = %s
bbox[lo] = %s
bbox[hi] = %s
center   = %s
maxprop  = %s
""" % (F.shape(),bb[0],bb[1],F.center(),F.maxProp())
    

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
    print("Reference Configuration: " + str(pyformex.refcfg))
    print("User Configuration: " + str(pyformex.cfg))
        

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
    In either case, the current dirctory is stored in pyformex.cfg['workdir']
    for persistence between pyFormex invocations.
    
    If fn does not exist, nothing is done.
    """
    if os.path.exists(fn):
        if not os.path.isdir(fn):
            fn = os.path.dirname(os.path.abspath(fn))
        os.chdir(fn)
        pyformex.cfg['workdir'] = fn
        pyformex.message("Your current workdir is %s" % os.getcwd())


def runtime():
    """Return the time elapsed since start of execution of the script."""
    return time.clock() - starttime


def startGui(args=[]):
    """Start the gui"""
    if pyformex.GUI is None:
        pyformex.debug("Starting the pyFormex GUI")
        from gui import guimain
        if guimain.startGUI(args) == 0:
            guimain.runGUI()


def isWritable(path):
    """Check that the specified path is writeable."""
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
        cur = int(utils.splitStartDigits(pyformex.__revision__.split()[1])[0])
        if not eval("%s %s %s" % (cur,comp,rev)):
            raise RuntimeError
    except:
        raise RuntimeError,"Your current pyFormex revision (%s) does not pass the test %s %s" % (pyformex.__revision__,comp,rev)
   
################### read and write files #################################

def writeGeomFile(filename,objects,sep=' ',mode='w'):
    """Save geometric objects to a pyFormex Geometry File.

    A pyFormex Geometry File can store multiple geometrical objects in a
    native format that can be efficiently read back into pyFormex.
    The format is portable over different pyFormex versions and 
    even to other software.

    -`filename`: the name of the file to be written
    -`objects`: a list or a dictionary. If it is a dictionary,
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
    native format that can be efficiently read back into pyFormex.
    The format is portable over different pyFormex versions and 
    even to other software.

    -`filename`: the name of an exisiting pyFormex Geometry File.
    
    Returns a dictionary with the geometric objects read from the file.
    If object names were stored in the file, they will be used as the keys.
    Else, default names will be provided.
    """
    f = geomfile.GeometryFile(filename,mode='r')
    return f.read()
    

#### End
