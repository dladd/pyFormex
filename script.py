#!/usr/bin/env python
# $Id$
"""Functions for executing pyFormex scripts."""

# THIS SHOULD ONLY CONTAIN FUNCTIONS INDEPENDENT FROM THE GUI

import globaldata as GD
import threading,os,commands,copy

import formex
import utils


######################### Exceptions #########################################

class Exit(Exception):
    """Exception raised to exit from a running script."""
    pass    

class ExitAll(Exception):
    """Exception raised to exit pyFormex from a script."""
    pass    

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
    print "pyFormex Error: "+message
    if not ack("Do you want to continue?"):
        exit()
    
def warning(message):
    print "pyFormex Warning: "+message
    if not ack("Do you want to continue?"):
        exit()

def info(message):
    print "pyFormex Info: "+message

def message(s):
    print s

########################### PLAYING SCRIPTS ##############################

scriptDisabled = False
scriptRunning = False
 
def playScript(scr):
    """Play a pyformex script scr. scr should be a valid Python text.

    There is a lock to prevent multiple scripts from being executed at the
    same time.
    """
    global scriptRunning, scriptDisabled, allowwait
    # (We only allow one script executing at a time!)
    # and scripts are non-reentrant
    if scriptRunning or scriptDisabled :
        return
    scriptRunning = True
    allowwait = True
    if GD.canvas:
        GD.canvas.update()
    if GD.gui:
        GD.gui.actions['Step'].setEnabled(True)
        GD.gui.actions['Continue'].setEnabled(True)
    # We need to pass formex globals to the script
    # This would be done automatically if we put this function
    # in the formex.py file. But then we need to pass other globals
    # from this file (like draw,...)
    # We might create a module with all operations accepted in
    # scripts.

    # Our solution is to take a copy of the globals in this module,
    # and add the globals from the 'formex' module
    # !! Taking a copy is needed to avoid changing this module's globals !!
    g = copy.copy(globals())
    g.update(formex.__dict__) # this also imports everything from numpy
    # Finally, we set the name to 'draw', so that the user can verify that
    # the script is executed from within the GUI.
    g.update({'__name__':'draw'})
    # Now we can execute the script using these collected globals
    
    exitall = False
    try:
        try:
            exec scr in g
        except Exit:
            pass
        except ExitAll:
            exitall = True
    finally:
        scriptRunning = False # release the lock in case of an error
        if GD.gui:
            GD.gui.actions['Step'].setEnabled(False)
            GD.gui.actions['Continue'].setEnabled(False)
    if exitall:
        exit()

def play(fn,name=None):
    """Play a formex script from file fn."""
    message("Running script (%s)" % fn)
    if name:
        GD.scriptName = name
    message("Running script (%s)" % fn)
    playScript(file(fn,'r'))
    message("Script finished")


def pause():
    pass

def step():
    pass

def fforward():
    pass


def listall():
    """List all Formices in globals()"""
    print "Formices currently in globals():"
    for n,t in globals().items():
        if isinstance(t,Formex):
            print "%s, " % n

def save(filename,fmt):
    pass


def system(cmdline,result='output'):
    if result == 'status':
        return os.system(cmdline)
    elif result == 'output':
        return commands.getoutput(cmdline)
    elif result == 'both':
        return commands.getstatusoutput(cmdline)

def exit(all=False):
    if scriptRunning:
        if all:
            raise ExitAll # exit from pyformex
        else:
            raise Exit # exit from script only
    else:
        sys.exit(0) # exit from pyformex

###########################  app  ################################


def runApp(args):
    """Run the application without gui."""
    # remaining args are interpreted as scripts
    for arg in args:
        if os.path.exists(arg) and utils.isPyFormex(arg):
            GD.scriptName = arg
            play(arg)


#### End
