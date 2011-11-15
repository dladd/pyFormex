#!/usr/bin/python
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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

# This is the only pyFormex module that is imported from the main script,
# so this is the place to put startup code

import sys,os
startup_warnings = ''
startup_messages = ''

pyformexdir = sys.path[0]
svnversion = os.path.exists(os.path.join(pyformexdir,'.svn'))
if svnversion:

    def checkLibraries():
        #print "Checking pyFormex libraries"
        msg = ''
        libdir = os.path.join(pyformexdir,'lib')
        libraries = [ 'misc_','nurbs_','drawgl_' ]
        for lib in libraries:
            src = os.path.join(libdir,lib+'module.c')
            obj = os.path.join(libdir,lib+'module.so')
            if not os.path.exists(obj) or os.path.getmtime(obj) < os.path.getmtime(src):
                msg += "\nThe compiled library '%smodule' is not up to date!" % lib
        return msg


    msg = checkLibraries()
    if msg:
        print "Rebuilding pyFormex libraries, please wait"
        cmd = "cd %s/lib;if [ ! -f Makefile ]; then ./configure; fi; make" % pyformexdir
        os.system(cmd)
        msg = checkLibraries()
    
    if msg:
        msg += """
        
I had a problem rebuilding the libraries in %s/lib. 
You should probably exit pyFormex, fix the problem first and then restart pyFormex.
""" % pyformexdir
    startup_warnings += msg

import utils

# intended Python version
minimal_version = '2.5'
target_version = '2.6'
found_version = utils.hasModule('python')


if utils.SaneVersion(found_version) < utils.SaneVersion(minimal_version):
#if utils.checkVersion('python',minimal_version) < 0:
    startup_warnings += """
Your Python version is %s, but pyFormex requires Python >= %s. We advice you to upgrade your Python version. Getting pyFormex to run on Python 2.4 requires only minor adjustements. Lower versions are problematic.
""" % (found_version,minimal_version)
    print startup_warnings
    sys.exit()
    
if utils.SaneVersion(found_version[:3]) > utils.SaneVersion(target_version):
#if utils.checkVersion('python',target_version) > 0:
    startup_warnings += """
Your Python version is %s, but pyFormex has only been tested with Python <= %s. We expect pyFormex to run correctly with your Python version, but if you encounter problems, please contact the developers at pyformex.berlios.de.
""" % (found_version,target_version)



import pyformex
from config import Config


# Remove unwanted warnings 
# We have moved this to the config file 
#utils.filterWarning('.*return_index.*','numpy')

###########################  main  ################################

def filterWarnings():
    try:
        for w in pyformex.cfg['warnings/filters']:
            utils.filterWarning(*w)
    except:
        pyformex.debug("Error while processing warning filters: %s" % pyformex.cfg['warnings/filters'])
    

def refLookup(key):
    """Lookup a key in the reference configuration."""
    try:
        return pyformex.refcfg[key]
    except:
        pyformex.debug("!There is no key '%s' in the reference config!"%key)
        return None


def prefLookup(key):
    """Lookup a key in the reference configuration."""
    return pyformex.prefcfg[key]
    

def printcfg(key):
    try:
        print("!! refcfg[%s] = %s" % (key,pyformex.refcfg[key]))
    except KeyError:
        pass
    print("!! cfg[%s] = %s" % (key,pyformex.cfg[key]))


def setRevision():
    sta,out = utils.runCommand('cd %s && svnversion' % pyformex.cfg['pyformexdir'],quiet=True)
    if sta == 0 and not out.startswith('exported'):
        pyformex.__revision__ = out.strip()


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

    # Create the user conf dir
    prefdir = os.path.dirname(pyformex.preffile)
    if not os.path.exists(prefdir):
        try:
            os.makedirs(prefdir)
        except:
            print("The path where your user preferences should be stored can not be created!\nPreferences are not saved!")
            return


    # Cleanup up the prefcfg
    del pyformex.prefcfg['__ref__']

    # Currently erroroneously processed, therefore not saved
    del pyformex.prefcfg['render']['light0']
    del pyformex.prefcfg['render']['light1']
    del pyformex.prefcfg['render']['light2']
    del pyformex.prefcfg['render']['light3']

    pyformex.options.debug = 1
    pyformex.debug("="*60)
    pyformex.debug("!!!Saving config:\n%s" % pyformex.prefcfg)
    
    try:
        pyformex.prefcfg.write(pyformex.preffile)
        res = "Saved"
    except:
        res = "Could not save"
    pyformex.debug("%s preferences to file %s" % (res,pyformex.preffile))


def apply_config_changes(cfg):
    """Apply incompatible changes in the configuration

    cfg is the user configuration that is to be saved.
    """
    # Safety checks
    if type(cfg['warnings/filters']) != list:
        cfg['warnings/filters'] = []
    
    # Adhoc changes
    if type(cfg['gui/dynazoom']) is str:
        cfg['gui/dynazoom'] = [ cfg['gui/dynazoom'], '' ]

    for i in range(8):
        t = "render/light%s"%i
        try:
            cfg[t] = dict(cfg[t])
        except:
            pass

    # Rename settings
    for old,new in [
        ('history','gui/history'),
        ]:
        if old in cfg.keys():
            if new not in cfg.keys():
                cfg[new] = cfg[old]
            del cfg[old]

    # Delete settings
    for key in [
        'input/timeout','filterwarnings',
        'render/ambient','render/diffuse','render/specular','render/emission',
        'render/material','canvas/propcolors','Save changes',
        ]:
        if key in cfg.keys():
            print("DELETING CONFIG VARIABLE %s" % key)
            del cfg[key]


def test_module(module):
    """Run the doctests in the modules docstrings."""
    import doctest
    # Note that a non-empty fromlist is needed to make the
    # __import__ function always return the imported module
    # even if a dotted path is specified
    mod = __import__(module,fromlist=['a'])
    return doctest.testmod(mod)


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
    import optparse
    from optparse import make_option as MO
    parser = optparse.OptionParser(
        # THE Qapp options are removed, because it does not seem to work !!!
        # SEE the comments in the gui.startGUI function  
        usage = "usage: %prog [<options>] [ [ scriptname [scriptargs] ] ...]",
        version = utils.FullVersion(),
        description = pyformex.Description,
        formatter = optparse.TitledHelpFormatter(),
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
           help="Go into interactive mode after processing the command line parameters. This is implied by the --gui option.",
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
        MO("--norst2html",
           action="store_false", dest="rst2html", default=True,
           help="Do not try to convert rst messages to html before displaying.",
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
        MO("--debug",
           action="store_const", dest="debug", const=-1,
           help="display debugging info to sys.stdout",
           ),
        MO("--debuglevel",
           action="store", dest="debug", type="int", default=0,
           help="display debugging info to sys.stdout",
           ),
        ## MO("--classify",
        ##    action="store_true", dest="classify", default=False,
        ##    help="classify the examples in categories",
        ##    ),
        MO("--newviewports",
           action="store_true", dest="newviewports", default=False,
           help="Use the new multiple viewport canvas implementation. This is an experimental feature only intended for developers.",
           ),
        MO("--testmodule",
           action="store", dest="testmodule", default=None,
           help="Run the docstring tests for module TESTMODULE. TESTMODULE is the name of the module, using . as path separator.",
           ),
        ## MO("--test",
        ##    action="store_true", dest="test", default=False,
        ##    help="testing mode: only for developers!",
        ##    ),
        MO("--testexecutor",
           action="store_true", dest="executor", default=False,
           help="test alternate executor: only for developers!",
           ),
        MO("--fastnurbs",
           action="store_true", dest="fastnurbs", default=False,
           help="test C library nurbs drawing: only for developers!",
           ),
        MO("--remove",
           action="store_true", dest="remove", default=False,
           help="remove the pyformex installation and exit",
           ),
        MO("--whereami",
           action="store_true", dest="whereami", default=False,
           help="show where the pyformex package is installed and exit",
           ),
        MO("--detect",
           action="store_true", dest="detect", default=False,
           help="show detected helper software and exit",
           ),
        ])
    pyformex.options, args = parser.parse_args(argv)
    pyformex.print_help = parser.print_help


    # process options
    if pyformex.options.nodefaultconfig and not pyformex.options.config:
        print("\nInvalid options: --nodefaultconfig but no --config option\nDo pyformex --help for help on options.\n")
        sys.exit()

    pyformex.debug("Options: %s" % pyformex.options)


    ########## Process special options which will not start pyFormex #######

    if pyformex.options.remove or \
       pyformex.options.whereami or \
       pyformex.options.detect or \
       pyformex.options.testmodule:

        if pyformex.options.remove:
            remove_pyFormex(pyformexdir,pyformex.scriptdir)

        if pyformex.options.whereami or pyformex.options.debug :
            print("Script started from %s" % pyformex.scriptdir)
            print("I found pyFormex in %s " %  pyformexdir)
            print("Current Python sys.path: %s" % sys.path)

        if pyformex.options.detect or pyformex.options.debug :
            print("Detecting all installed helper software")
            utils.checkExternal()
            print(utils.reportDetected())

        if pyformex.options.testmodule:
            for a in pyformex.options.testmodule.split(','):
                test_module(a)

        sys.exit()

    ########### Read the config files  ####################

    # These values should not be changed
    pyformex.cfg.userprefs = os.path.join(pyformex.cfg.userconfdir,'pyformexrc')
    pyformex.cfg.autorun = os.path.join(pyformex.cfg.userconfdir,'startup.py')
    
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
    pyformex.debug("="*60)
    pyformex.debug("RefConfig: %s" % pyformex.refcfg)

    # Use the last as place to save preferences
    pyformex.prefcfg = Config(default=refLookup)
    if os.path.exists(pyformex.preffile):
        pyformex.debug("Reading config file %s" % pyformex.preffile)
        pyformex.prefcfg.read(pyformex.preffile)
    pyformex.debug("="*60)
    pyformex.debug("Config: %s" % pyformex.prefcfg)

    # Fix incompatible changes in configuration
    apply_config_changes(pyformex.prefcfg)

    # Create an empty one for the session settings
    pyformex.cfg = Config(default=prefLookup)

    # This should probably be changed to options overriding config
    # Set option from config if it was not explicitely given
    if pyformex.options.uselib is None:
        pyformex.options.uselib = pyformex.cfg['uselib']


    # Set default --nogui if first remaining argument is a pyformex script.
    if pyformex.options.gui is None:
        pyformex.options.gui = not (len(args) > 0 and utils.is_pyFormex(args[0]))

    if pyformex.options.gui:
        pyformex.options.interactive = True

    # Set Revision and run svnclean if we run from an SVN version
    if svnversion:
        setRevision()
        svnclean = os.path.join(pyformexdir,'svnclean')
        if os.path.exists(svnclean):
            try:
                utils.runCommand(svnclean)
            except:
                print("Error while executing %s, we ignore it and continue" % svnclean)

        def getSVNURL():
            sta,out = utils.runCommand("cd %s;svn info | grep -F 'URL:'"%pyformexdir)
            if sta == 0:
                return out
            else:
                return ''


        ## s = getSVNURL()
        ## print s
        ## import re
        ## m = re.match(".*//(?P<user>[^@]*)@svn\.berlios\.de.*",s)
        ## pyformex.svnuser = m.group('user')
        ## print pyformex.svnuser

        # Add in subversion specific config
        devhowto = os.path.join(pyformexdir,'..','HOWTO-dev.rst')
        builddoc = os.path.join(pyformexdir,"doc","build-local-docs.rst")
        pyformex.refcfg.help['developer'][0:0] = [('Developer HOWTO',devhowto),('&Build local documentation',builddoc)]
        #print pyformex.refcfg.help['developer']
    


    ###### We have the config and options all set up ############
    filterWarnings()



    def _format_warning(message,category,filename,lineno,line=None):
        """Replace the default warnings.formatwarning

        This allows the warnings being called using a simple mnemonic
        string. The full message is then found from the message module.
        """
        import messages
        message = messages.getMessage(message)
        message = """..

pyFormex Warning
================
%s

`Called from:` %s `line:` %s
""" % (message,filename,lineno)
        if line:
            message += "%s\n" % line
        return message


    if pyformex.cfg['warnings/nice']:
        import warnings
        warnings.formatwarning = _format_warning
    


    # Start the GUI if needed
    # Importing the gui should be done after the config is set !!
    if pyformex.options.gui:
        from gui import guimain
        pyformex.debug("GUI version")
        res = guimain.startGUI(args)
        if res != 0:
            print("Could not start the pyFormex GUI: %s" % res)
            return res # EXIT

    # Display the startup warnings and messages
    if startup_warnings:
        if pyformex.cfg['startup_warnings']:
            pyformex.warning(startup_warnings)
        else:
            print(startup_warnings)
    if startup_messages:
        print(startup_messages)

    if pyformex.options.debug: # Avoid computing the report if not printed
        pyformex.debug(utils.reportDetected())
 
    #print(pyformex.cfg.keys())
    #print(pyformex.refcfg.keys())
      
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
                return res # EXIT
                
    else:
        pyformex.debug("stdin is a tty: %s" % sys.stdin.isatty())
        # Play script from stdin
        # Can we check for interactive session: stdin connected to terminal?
        #from script import playScript
        #playScript(sys.stdin)
        

    # after processing all args, go into interactive mode
    if pyformex.options.gui and pyformex.app:
        res = guimain.runGUI()

    ## elif pyformex.options.interactive:
    ##     print("Enter your script and end with CTRL-D")
    ##     from script import playScript
    ##     playScript(sys.stdin)
        
    #Save the preferences that have changed
    savePreferences()

    # Exit
    return res


# End
