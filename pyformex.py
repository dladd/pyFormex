#!/usr/bin/env python
# $Id$
"""pyformex is a python implementation of Formex algebra"""

import globaldata as GD

import sys
import os.path
from config import Config
from optparse import OptionParser,make_option


###########################  main  ################################


def refLookup(key):
    """Lookup a key in the reference configuration."""
    return GD.refcfg[key]


def printcfg(key):
    try:
        print "!! refcfg[%s] = %s" % (key,GD.refcfg[key])
    except KeyError:
        pass
    print "!! cfg[%s] = %s" % (key,GD.cfg[key])
    

def main(argv=None):
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
       user settings:        $HOME/.pyformexrc
       local settings        $PWD/.pyformexrc
    Also, an extra config file can be specified in the command line.
    Config file settings always override previous ones.
    On exit, the preferences that were changed are written to the last
    read config file. Changed settings are those that differ from the settings
    in all but the last one.
    """
    # this allows us to call main from the interpreter
    if argv is None:
        argv = sys.argv
        
    # get/set the path to the pyformex files, and store it in the config
    pyformexdir = os.path.dirname(os.path.realpath(argv[0]))
    # use a read, not an update, to set the pyformexdir as a variable
    GD.cfg = Config()
    GD.cfg.read("pyformexdir = '%s'\n" % pyformexdir)

    # get/set the user's home dir
    if os.name == 'posix':
        homedir = os.environ['HOME']
    elif os.name == 'nt':
        homedir = os.environ['HOMEDRIVE']+os.environ['HOMEPATH']
    GD.cfg.read("homedir = '%s'\n" % homedir)

    # Process options
    parser = OptionParser(
        usage = "usage: %prog [<options>] [ --  <Qapp-options> ]",
        version = GD.Version,
        option_list=[
        make_option("--dri", help="Force the use of Direct Rendering",
                    action="store_true", dest="dri", default=False),
        make_option("--nodri", help="Disables the use of Direct Rendering (overrides the --dri options)",
                    action="store_true", dest="nodri", default=False),
        make_option("--makecurrent", help="Call makecurrent on initializing the OpenGL canvas",
                    action="store_true", dest="makecurrent", default=False),
        make_option("--nogui", help="do not load the GUI",
                    action="store_false", dest="gui", default=True),
        make_option("--config", help="use file CONFIG for settings",
                    action="store", dest="config", default=None),
         make_option("--nodefaultconfig", help="skip all default locations of config files",
                    action="store_true", dest="nodefaultconfig", default=False),
       make_option("--redirect", help="redirect standard output to the message board (ignored with --nogui)",
                    action="store_true", dest="redirect", default=False),
       make_option("--multiview", help="Activate the multiple viewport feature (ignored with --nogui)",
                    action="store_true", dest="multiview", default=False),
       make_option("--debug", help="display debugging info to sys.stdout",
                    action="store_true", dest="debug", default=False),
        ])
    GD.options, args = parser.parse_args()

    GD.debug("Options: %s" % GD.options)

    # Read the config files
    defaults = os.path.join(pyformexdir,"pyformexrc")
    if os.name == 'posix':
        siteprefs = '/etc/pyformexrc'
        prefs = '.pyformexrc'
    elif os.name == 'nt':
        siteprefs = None  # Where does Windows put site prefs?
        prefs = 'pyformex.cfg'
    homeprefs = os.path.join(homedir,prefs)
    localprefs = os.path.join(os.getcwd(),prefs)

    sysprefs = filter(os.path.exists,[defaults,siteprefs])
    userprefs = filter(os.path.exists,[homeprefs,localprefs])
    if GD.options.nodefaultconfig:
        sysprefs = sysprefs[0:1]
        userprefs = []
    if GD.options.config:
        userprefs.append(GD.options.config)
    if len(userprefs) == 0:
        userprefs = [homeprefs]
    allprefs = sysprefs + userprefs
    GD.preffile = allprefs.pop()
    for f in allprefs:
        GD.debug("Reading config file %s" % f)
        GD.cfg.read(f)
    # Save this config as a reference, then load last config file
    GD.refcfg = GD.cfg
    GD.cfg = Config(default=refLookup)
    if os.path.exists(GD.preffile):
        GD.debug("Reading config file %s" % GD.preffile)
        GD.cfg.read(GD.preffile)
    GD.debug("RefConfig: %s" % GD.refcfg)
    GD.debug("Config: %s" % GD.cfg)

    # Run the application with the remaining arguments
    # Importing the gui should be done after the config is set !!
    if GD.options.gui:
        from gui.gui import runApp
    else:
        from script import runApp
    
    res = runApp(args)

    #Save the preferences that have changed
    GD.savePreferences()

    # Exit
    return res


#### Go

if __name__ == "__main__":
    sys.exit(main())

#### End
