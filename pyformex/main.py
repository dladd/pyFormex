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

import sys,os
pyformexdir = sys.path[0]
svnversion = os.path.exists(os.path.join(pyformexdir,'.svn'))

import pyformex
from pyformex import *

import warnings
warnings.filterwarnings('ignore','.*return_index.*',UserWarning,'numpy')


import utils
from config import Config

###########################  main  ################################

def refLookup(key):
    """Lookup a key in the reference configuration."""
    return pyformex.refcfg[key]


def printcfg(key):
    try:
        print("!! refcfg[%s] = %s" % (key,pyformex.refcfg[key]))
    except KeyError:
        pass
    print("!! cfg[%s] = %s" % (key,pyformex.cfg[key]))


def setRevision():
    sta,out = utils.runCommand('cd %s && svnversion' % pyformex.cfg['pyformexdir'],quiet=True)
    if sta == 0 and not out.startswith('exported'):
        pyformex.__revision__ = "$Rev: %s $" % out.strip()



def remove_pyFormex(pyformexdir,scriptdir):
    """Remove the pyFormex installation."""
    print("""
BEWARE!
This procedure will remove the complete pyFormex installation!
If you continue, pyFormex will exit and you will not be able to run it again.
The pyFormex installation is in: %s
The pyFormex executable script is in: %s
You will need proper permissions to actually delete the files.
""" % (pyformexdir,scriptdir))
    s = raw_input("Are you sure you want to remove pyFormex? yes/NO: ")
    if s == 'yes':
        print("Removing %s" % pyformexdir)
        utils.removeTree(pyformexdir)
        script = os.path.join(scriptdir,'pyformex')
        egginfo = "%s-%s.egg-info" % (pyformexdir,pyformex.__version__.replace('-','_'))
        for f in [ script,egginfo ]:
            if os.path.exists(f):
                print("Removing %s" % f)
                os.remove(f)
        print("\nBye, bye! I won't be back until you reinstall me!")
    elif s.startswith('y') or s.startswith('Y'):
        print("You need to type exactly 'yes' to remove me.")
    else:
        print("Thanks for letting me stay this time.")
    sys.exit()


def savePreferences():
    """Save the preferences.

    The name of the preferences file is determined at startup from
    the configuration files, and saved in ``pyformex.preffile``.
    If a local preferences file was read, it will be saved there.
    Otherwise, it will be saved as the user preferences, possibly
    creating that file.
    If ``pyformex.preffile`` is None, preferences are not saved.
    """
    if pyformex.preffile is None:
        return
    
    del pyformex.cfg['__ref__']

    # Dangerous to set permanently!
    del pyformex.cfg['input/timeout']
    
    debug("!!!Saving config:\n%s" % pyformex.cfg)

    try:
        fil = file(pyformex.preffile,'w')
        fil.write("%s" % pyformex.cfg)
        fil.close()
        res = "Saved"
    except:
        res = "Could not save"
    debug("%s preferences to file %s" % (res,pyformex.preffile))

###########################  app  ################################

    
def run(argv=[]):
    """This is a fairly generic main() function.

    It is responsible for reading the configuration file(s),
    processing the command line options and starting the application.
    The basic configuration file is 'pyformexrc' located in the pyformex
    directory. It should always be present and be left unchanged.
    You can copy this file to another location if you want to make changes.
    By default, pyformex will try to read the following extra configuration
    files (in this order:
        default settings:     <pyformexdir>/pyformexrc
        system-wide settings: /etc/pyformexrc
        user settings:        $HOME/.pyformex/pyformexrc
        local settings        $PWD/.pyformexrc
    Also, an extra config file can be specified in the command line.
    Config file settings always override previous ones.
    On exit, the preferences that were changed are written to the last
    read config file. Changed settings are those that differ from the settings
    in all but the last one.
    """
    # Create a config instance
    pyformex.cfg = Config()
    # Fill in the pyformexdir and homedir variables
    # (use a read, not an update)
    if os.name == 'posix':
        homedir = os.environ['HOME']
    elif os.name == 'nt':
        homedir = os.environ['HOMEDRIVE']+os.environ['HOMEPATH']
    pyformex.cfg.read("pyformexdir = '%s'\n" % pyformexdir)
    pyformex.cfg.read("homedir = '%s'\n" % homedir)

    # Read the defaults (before the options)
    defaults = os.path.join(pyformexdir,"pyformexrc")
    pyformex.cfg.read(defaults)
    
    # Process options
    from optparse import OptionParser,make_option as MO
    parser = OptionParser(
        usage = "usage: %prog [<options>] [ --  <Qapp-options> ] [[ scriptname [scriptargs]] ...]",
        version = pyformex.Version,
        option_list=[
        MO("--gui",
           action="store_true", dest="gui", default=None,
           help="start the GUI (default if no scriptfile argument is given)",
           ),
        MO("--nogui",
           action="store_false", dest="gui", default=None,
           help="do not load the GUI (default if a scriptfile argument is given)",
           ),
        MO("--interactive",'-i',
           action="store_true", dest="interactive", default=False,
           help="go into interactive mode after processing the command line parameters. This is implied by the --gui option.",
           ),
        MO("--force-dri",
           action="store_true", dest="dri", default=None,
           help="Force use of Direct Rendering",
           ),
        MO("--force-nodri",
           action="store_false", dest="dri", default=None,
           help="Disables the Direct Rendering",
           ),
        MO("--uselib",
           action="store_true", dest="uselib", default=None,
           help="Use the pyFormex C lib if available. This is the default.",
           ),
        MO("--nouselib",
           action="store_false", dest="uselib", default=None,
           help="Do not use the pyFormex C-lib.",
           ),
        MO("--safelib",
           action="store_true", dest="safelib", default=True,
           help="Convert data types to match C-lib. This is the default.",
           ),
        MO("--unsafelib",
           action="store_false", dest="safelib", default=True,
           help="Do not convert data types to match C-lib. BEWARE: this may make the C-lib calls impossible. Use only for debugging purposes.",
           ),
        MO("--fastencode",
           action="store_true", dest="fastencode", default=False,
           help="Use a fast algorithm to encode edges.",
           ),
        MO("--config",
           action="store", dest="config", default=None,
           help="Use file CONFIG for settings",
           ),
        MO("--nodefaultconfig",
           action="store_true", dest="nodefaultconfig", default=False,
           help="Skip the default site and user config files. This option can only be used in conjunction with the --config option.",
           ),
        MO("--redirect",
           action="store_true", dest="redirect", default=False,
           help="Redirect standard output to the message board (ignored with --nogui)",
           ),
        MO("--detect",
           action="store_true", dest="detect", default=False,
           help="Detect helper software and print report.",
           ),
        MO("--debug",
           action="store_true", dest="debug", default=False,
           help="display debugging info to sys.stdout",
           ),
        MO("--classify",
           action="store_true", dest="classify", default=False,
           help="classify the examples in categories",
           ),
        MO("--whereami",
           action="store_true", dest="whereami", default=False,
           help="show where the pyformex package is located",
           ),
        MO("--remove",
           action="store_true", dest="remove", default=False,
           help="remove the pyformex installation",
           ),
        MO("--test",
           action="store_true", dest="test", default=False,
           help="testing mode: only for developers!",
           ),
        MO("--testfuse",
           action="store_true", dest="testfuse", default=False,
           help="test new fuse algorithm: only for developers!",
           ),
        MO("--testhighlight",
           action="store_true", dest="testhighlight", default=False,
           help="highlight testing mode: only for developers!",
           ),
        MO("--executor",
           action="store_true", dest="executor", default=False,
           help="test alternate executor: only for developers!",
           ),
        ])
    pyformex.options, args = parser.parse_args(argv)
    pyformex.print_help = parser.print_help


    # process options
    if pyformex.options.nodefaultconfig and not pyformex.options.config:
        print("\nInvalid options: --nodefaultconfig but no --config option\nDo pyformex --help for help on options.\n")
        sys.exit()


    if pyformex.options.whereami:
        print("Script started from %s" % pyformex.scriptdir)
        print("I found pyFormex in %s " %  pyformexdir)
        print("Current Python sys.path: %s" % sys.path)
        sys.exit()
        
    if pyformex.options.remove:
        remove_pyFormex(pyformexdir,pyformex.scriptdir)
        
    if pyformex.options.detect:
        print("Detecting all installed helper software")
        utils.checkExternal()
        print(utils.reportDetected())
        sys.exit()
        
    pyformex.debug("Options: %s" % pyformex.options)

    ########### Read the config files  ####################

    # Create the user conf dir
    if not os.path.exists(pyformex.cfg.userconfdir):
        os.mkdir(pyformex.cfg.userconfdir)

    # These values  should not be changed
    pyformex.cfg.userprefs = os.path.join(pyformex.cfg.userconfdir,'pyformexrc')
    pyformex.cfg.autorun = os.path.join(pyformex.cfg.userconfdir,'startup.py')

    # Migrate old user prefs
    olduserprefs = os.path.join(pyformex.cfg.homedir,'.pyformexrc')
    if not os.path.exists(pyformex.cfg.userprefs) and os.path.exists(olduserprefs):
        import shutil
        print("Moving user preferences to new location")
        print("%s --> %s" % (olduserprefs,pyformex.cfg.userprefs))
        shutil.move(olduserprefs,pyformex.cfg.userprefs)
    
    # Set the config files
    if pyformex.options.nodefaultconfig:
        sysprefs = []
        userprefs = []
    else:
        sysprefs = [ pyformex.cfg.siteprefs ]
        userprefs = [ pyformex.cfg.userprefs ]
        if os.path.exists(pyformex.cfg.localprefs):
            userprefs.append(pyformex.cfg.localprefs)


    if pyformex.options.config:
        userprefs.append(pyformex.options.config)

    if len(userprefs) == 0:
        # We should always have a place to store the user preferences
        userprefs = [ pyformex.cfg.userprefs ]

    pyformex.preffile = os.path.abspath(userprefs[-1]) # Settings will be saved here
   
    # Read all but the last as reference
    for f in filter(os.path.exists,sysprefs + userprefs[:-1]):
        pyformex.debug("Reading config file %s" % f)
        pyformex.cfg.read(f)
    pyformex.refcfg = pyformex.cfg
    pyformex.debug("RefConfig: %s" % pyformex.refcfg)

    # Use the last as place to save preferences
    pyformex.cfg = Config(default=refLookup)
    if os.path.exists(pyformex.preffile):
        pyformex.debug("Reading config file %s" % pyformex.preffile)
        pyformex.cfg.read(pyformex.preffile)
    pyformex.debug("Config: %s" % pyformex.cfg)

    # Set option from config if it was not explicitely given
    if pyformex.options.uselib is None:
        pyformex.options.uselib = pyformex.cfg['uselib'] 

    # Set default --nogui if first remaining argument is a pyformex script.
    if pyformex.options.gui is None:
        pyformex.options.gui = not (len(args) > 0 and utils.isPyFormex(args[0]))

    if pyformex.options.gui:
        pyformex.options.interactive = True

    # Set Revision if we run from an SVN version
    if svnversion:
        setRevision()

    pyformex.debug(utils.reportDetected())

    #print("pyformex",utils.subDict(pyformex.cfg,'canvas/'))
    #print("DEFAULT",self.default)
    #print(pyformex.cfg.keys())
    #print(pyformex.refcfg.keys())
    
    # Start the GUI if needed
    # Importing the gui should be done after the config is set !!
    if pyformex.options.gui:
        pyformex.debug("GUI version")
        from gui import gui
        gui.startGUI(args)
    #
    # Qt4 may have changed the locale.
    # Since a LC_NUMERIC setting other than C may cause lots of troubles
    # with reading and writing files (formats become incompatible!)
    # we put it back to a sane setting
    #
    utils.setSaneLocale()

    # Initialize the libraries
    #print("NOW LOAIDNG LIBS")
    #import lib
    #lib.init_libs(pyformex.options.uselib,pyformex.options.gui)


    # Prepend the autorun scripts
    ar = pyformex.cfg.get('autorun','')
    if ar :
        if type(ar) is str:
            ar = [ ar ]
        # expand tilde, as would bash
        ar = map(utils.tildeExpand,ar)
        args[0:0] = [ fn for fn in ar if os.path.exists(fn) ]

    # remaining args are interpreted as scripts and their parameters
    res = 0
    if args:
        pyformex.debug("Remaining args: %s" % args)
        from script import processArgs
        res = processArgs(args)
        
        if res:
            if pyformex.options.gui:
                pyformex.message("There was an error while executing a script")
            else:
                return res
                
    else:
        pyformex.debug("stdin is a tty: %s" % sys.stdin.isatty())
        # Play script from stdin
        # Can we check for interactive session: stdin connected to terminal?
        #from script import playScript
        #playScript(sys.stdin)
        

    # after processing all args, go into interactive mode
    if pyformex.options.gui:
        res = gui.runGUI()

    ## elif pyformex.options.interactive:
    ##     print("Enter your script and end with CTRL-D")
    ##     from script import playScript
    ##     playScript(sys.stdin)
        
    #Save the preferences that have changed
    savePreferences()

    # Exit
    return res


# End
