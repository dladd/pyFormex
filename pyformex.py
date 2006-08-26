#!/usr/bin/env python
# $Id$
"""pyformex is a python implementation of Formex algebra"""

import globaldata as GD

import sys
import os.path
from optparse import OptionParser,make_option


###########################  main  ################################


def main(argv=None):
    """This is a fairly generic main() function.

    It is responsible for reading the configuration file(s),
    processing the command line options and starting the application.
    The basic configuration file is 'pyformexrc' located in the pyformex
    directory. It should always be present and be left unchanged.
    You can copy this file to another location if you want to make changes.
    By default, pyformex will try to read the following extra configuration
    files (in this order:
       system-wide settings: /etc/pyformexrc
       user settings:        $HOME/.pyformexrc
       local settings        $PWD/.pyformexrc
    Also, an extra config file can be specified in the command line.
    Config file settings always override existing ones.
    """
    # this allows us to call main from the interpreter
    if argv is None:
        argv = sys.argv
    # get the path to the pyformex files, and store it in the config
    pyformexdir = os.path.dirname(os.path.realpath(argv[0]))
    # use a read, not an update, to set the pyformexdir as a variable
    GD.cfg.read("pyformexdir = '%s'\n" % pyformexdir)
    if os.name == 'posix':
        homedir = os.environ['HOME']
        prefs = '.pyformex'
        siteprefs = '/etc/pyformex'
    elif os.name == 'nt':
        homedir = os.environ['HOMEDRIVE']+os.environ['HOMEPATH']
        prefs = 'pyformex.cfg'
        siteprefs = None  # Where does Windows put site prefs?
    GD.prefs = prefs
    GD.userprefs = os.path.join(homedir,prefs)
    GD.cfg.read("homedir = '%s'\n" % homedir)
    
    # Read the default (distribution) config file
    GD.cfg.read(os.path.join(pyformexdir,"pyformexrc"))
    # Read additional config files
    for f in [ siteprefs, GD.userprefs, GD.prefs ]:
        if f and os.path.exists(f):
            print f
            GD.cfg.read(f)

    # HACK !!
    # set pyformexdir back to the built-in
    #
    GD.cfg.pyformexdir = pyformexdir
    GD.cfg.icondir = os.path.join(GD.cfg.pyformexdir,'icons')
    GD.cfg.exampledir = os.path.join(GD.cfg.pyformexdir,'examples')


    # Process options
    parser = OptionParser(
        usage = "usage: %prog [<options>] [ --  <Qapp-options> ]",
        version = GD.Version,
        option_list=[
        make_option("--nodri", help="do not use Direct Rendering",
                    action="store_false", dest="dri", default=True),
        make_option("--nogui", help="do not load the GUI",
                    action="store_false", dest="gui", default=True),
        make_option("--nosplash", help="do not show the startup screen",
                    action="store_false", dest="splash", default=True),
        make_option("--debug", help="display logging info to sys.stdout",
                    action="store_true", dest="debug", default=False)
        ])
    GD.options, args = parser.parse_args()

    if GD.options.debug:
        print "Options: ",GD.options
        print "Config: ",GD.cfg

    # Run the application with the remaining arguments
    
    # Importing the gui should be done after the config is set !!
    if GD.options.gui:
        from gui.gui import runApp
    else:
        from script import runApp
    
    res = runApp(args)

    # Exit
    return res


#### Go

if __name__ == "__main__":
    sys.exit(main())

#### End
