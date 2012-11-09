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

# This is the only pyFormex module that is imported by the main script,
# so this is the place to put startup code
"""pyFormex main module

This module contains the main function of pyFormex, which is run by the
startup script.
"""
from __future__ import print_function
import pyformex as pf

import sys,os
startup_warnings = ''
startup_messages = ''

pyformexdir = sys.path[0]

if os.path.exists(os.path.join(pyformexdir,'.svn')):
    # Running from source tree
    pf.installtype = 'S'

    def checkLibraries():
        #print "Checking pyFormex libraries"
        msg = ''
        libdir = os.path.join(pyformexdir,'lib')
        libraries = [ 'misc_','nurbs_','drawgl_' ]
        for lib in libraries:
            src = os.path.join(libdir,lib+'.c')
            obj = os.path.join(libdir,lib+'.so')
            if not os.path.exists(obj) or os.path.getmtime(obj) < os.path.getmtime(src):
                msg += "\nThe compiled library '%s' is not up to date!" % lib
        return msg


    msg = checkLibraries()
    if msg:
        print("Rebuilding pyFormex libraries, please wait")
        cmd = "cd %s/..; make lib" % pyformexdir
        os.system(cmd)
        msg = checkLibraries()
    
    if msg:
        msg += """
        
I had a problem rebuilding the libraries in %s/lib. 
You should probably exit pyFormex, fix the problem first and then restart pyFormex.
""" % pyformexdir
    startup_warnings += msg


import utils

# Set the proper revision number when running from svn sources
if pf.installtype=='S':
    try:
        sta,out = utils.runCommand('cd %s && svnversion' % pyformexdir,quiet=True)
        if sta == 0 and not out.startswith('exported'):
            pf.__revision__ = out.strip()
    except:
        # The above may fail when a checked-out svn version is moved to
        # a system without subversion installed.
        # Therefore, silently ignore
        pass

# Set the Full pyFormex version string
# This had to be deferred until the __revision__ was set
pf.FullVersion = '%s (Rev. %s)' % (pf.Version,pf.__revision__)

# intended Python version
minimal_version = '2.5'
target_version = '2.7'
found_version = utils.hasModule('python')


if utils.SaneVersion(found_version) < utils.SaneVersion(minimal_version):
#if utils.checkVersion('python',minimal_version) < 0:
    startup_warnings += """
Your Python version is %s, but pyFormex requires Python >= %s. We advice you to upgrade your Python version. Getting pyFormex to run on Python 2.4 requires only minor adjustements. Lower versions are problematic.
""" % (found_version,minimal_version)
    print(startup_warnings)
    sys.exit()
    
if utils.SaneVersion(found_version[:3]) > utils.SaneVersion(target_version):
#if utils.checkVersion('python',target_version) > 0:
    startup_warnings += """
Your Python version is %s, but pyFormex has only been tested with Python <= %s. We expect pyFormex to run correctly with your Python version, but if you encounter problems, please contact the developers at http://pyformex.org.
""" % (found_version,target_version,)



from config import Config

###########################  main  ################################

def filterWarnings():
    pf.debug("Current warning filters: %s" % pf.cfg['warnings/filters'],pf.DEBUG.WARNING)
    try:
        for w in pf.cfg['warnings/filters']:
            utils.filterWarning(*w)
    except:
        pf.debug("Error while processing warning filters: %s" % pf.cfg['warnings/filters'],pf.DEBUG.WARNING)
    

def refLookup(key):
    """Lookup a key in the reference configuration."""
    try:
        return pf.refcfg[key]
    except:
        pf.debug("!There is no key '%s' in the reference config!"%key,pf.DEBUG.CONFIG)
        return None


def prefLookup(key):
    """Lookup a key in the reference configuration."""
    return pf.prefcfg[key]
    

def printcfg(key):
    try:
        print("!! refcfg[%s] = %s" % (key,pf.refcfg[key]))
    except KeyError:
        pass
    print("!! cfg[%s] = %s" % (key,pf.cfg[key]))


def remove_pyFormex(pyformexdir,bindir):
    """Remove the pyFormex installation."""
    if pf.installtype == 'P':
        print("It looks like this version of pyFormex was installed from a distribution package. You should use your distribution's package tools to remove the pyFormex installation.")
        return
    
    if pf.installtype == 'S':
        print("It looks like you are running pyFormex directly from a source tree at %s. I will not remove it. If you have enough privileges, you can just remove the whole source tree from the file system." % pyformexdir)
        return
    
        
    print("""
BEWARE!
This procedure will remove the complete pyFormex installation!
You should only use this on a pyFormex installed with 'python setup.py install'.
If you continue, pyFormex will exit and you will not be able to run it again.
The pyFormex installation is in: %s
The pyFormex executable script is in: %s
You will need proper permissions to actually delete the files.
""" % (pyformexdir,bindir))
    s = raw_input("Are you sure you want to remove pyFormex? yes/NO: ")
    if s == 'yes':
        print("Removing %s" % pyformexdir)
        utils.removeTree(pyformexdir)
        script = os.path.join(bindir,'pyformex')
        egginfo = "%s-%s.egg-info" % (pyformexdir,pf.__version__.replace('-','_'))
        datadir = os.path.commonprefix(['/usr/local/share',pyformexdir])
        datadir = os.path.join(datadir,'share')
        data = utils.prefixFiles(datadir,['man/man1/pyformex.1',
                                          'applications/pyformex.desktop',
                                          'pixmaps/pyformex-64x64.png',
                                          'pixmaps/pyformex.xpm'])
        for f in [ script,egginfo ] + data:
            if os.path.exists(f):
                print("Removing %s" % f)
                os.remove(f)
            else:
                print("Could not remove %s" % f)
        
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
    if pf.preffile is None:
        return

    # Create the user conf dir
    prefdir = os.path.dirname(pf.preffile)
    if not os.path.exists(prefdir):
        try:
            os.makedirs(prefdir)
        except:
            print("The path where your user preferences should be stored can not be created!\nPreferences are not saved!")
            return


    # Cleanup up the prefcfg
    del pf.prefcfg['__ref__']

    # Currently erroroneously processed, therefore not saved
    del pf.prefcfg['render']['light0']
    del pf.prefcfg['render']['light1']
    del pf.prefcfg['render']['light2']
    del pf.prefcfg['render']['light3']

    pf.debug("="*60,pf.DEBUG.CONFIG)
    pf.debug("!!!Saving config:\n%s" % pf.prefcfg,pf.DEBUG.CONFIG)
    
    try:
        pf.prefcfg.write(pf.preffile)
        res = "Saved"
    except:
        res = "Could not save"
    pf.debug("%s preferences to file %s" % (res,pf.preffile),pf.DEBUG.CONFIG)


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

    for d in [ 'scriptdirs', 'appdirs' ]:
        if d in cfg:
            scriptdirs = []
            for i in cfg[d]:
                if i[1] == '' or os.path.isdir(i[1]):
                    scriptdirs.append(tuple(i))
                elif i[0] == '' or os.path.isdir(i[0]):
                    scriptdirs.append((i[1],i[0]))
            cfg[d] = scriptdirs

    # Rename settings
    for old,new in [
        ('history','gui/scripthistory'),
        ('gui/history','gui/scripthistory'),
        ('raiseapploadexc','showapploaderrors'),
        ]:
        if old in cfg.keys():
            if new not in cfg.keys():
                cfg[new] = cfg[old]
            del cfg[old]

    # Delete settings
    for key in [
        'input/timeout','filterwarnings',
        'render/ambient','render/diffuse','render/specular','render/emission',
        'render/material','canvas/propcolors','Save changes','canvas/bgmode',
        'canvas/bgcolor2',
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
    pf.cfg = Config()
    # Fill in the pyformexdir and homedir variables
    # (use a read, not an update)
    if os.name == 'posix':
        homedir = os.environ['HOME']
    elif os.name == 'nt':
        homedir = os.environ['HOMEDRIVE']+os.environ['HOMEPATH']
    pf.cfg.read("pyformexdir = '%s'\n" % pyformexdir)
    pf.cfg.read("homedir = '%s'\n" % homedir)

    # Read the defaults (before the options)
    defaults = os.path.join(pyformexdir,"pyformexrc")
    pf.cfg.read(defaults)
    
    # Process options
    import optparse
    from optparse import make_option as MO
    parser = optparse.OptionParser(
        # THE Qapp options are removed, because it does not seem to work !!!
        # SEE the comments in the gui.startGUI function  
        usage = "usage: %prog [<options>] [ [ scriptname [scriptargs] ] ...]",
        version = utils.FullVersion(),
        description = pf.Description,
        formatter = optparse.TitledHelpFormatter(),
        option_list=[
        MO("--gui",
           action="store_true", dest="gui", default=None,
           help="Start the GUI (this is the default when no scriptname argument is given)",
           ),
        MO("--nogui",
           action="store_false", dest="gui", default=None,
           help="Do not start the GUI (this is the default when a scriptname argument is given)",
           ),
        MO("--interactive",
           action="store_true", dest="interactive", default=False,
           help="Go into interactive mode after processing the command line parameters. This is implied by the --gui option.",
           ),
        MO("--dri",
           action="store_true", dest="dri", default=None,
           help="Use Direct Rendering Infrastructure. By default, direct rendering will be used if available.",
           ),
        MO("--nodri",
           action="store_false", dest="dri", default=None,
           help="Do not use the Direct Rendering Infrastructure. This may be used to turn off the direc rendering, e.g. to allow better capturing of images and movies.",
           ),
        MO("--uselib",
           action="store_true", dest="uselib", default=None,
           help="Use the pyFormex C lib if available. This is the default.",
           ),
        MO("--nouselib",
           action="store_false", dest="uselib", default=None,
           help="Do not use the pyFormex C-lib.",
           ),
        MO("--commands",
           action="store_true", dest="commands", default=False,
           help="Use the commands module to execute external commands. Default is to use the subprocess module.",
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
           action="store_true", dest="redirect", default=None,
           help="Redirect standard output to the message board (ignored with --nogui)",
           ),
        MO("--noredirect",
           action="store_false", dest="redirect",
           help="Do not redirect standard output to the message board.",
           ),
        MO("--debug",
           action="store", dest="debug", default='',
           help="Display debugging information to sys.stdout. The value is a comma-separated list of (case-insensitive) strings corresponding with the attributes of the DebugLevels class. The individual values are OR-ed together to produce a final debug value. The special value 'all' can be used to switch on all debug info.",
           ),
        MO("--debuglevel",
           action="store", dest="debuglevel", type="int", default=0,
           help="Display debugging info to sys.stdout. The value is an int with the bits of the requested debug levels set. A value of -1 switches on all debug info. If this option is used, it overrides the --debug option.",
           ),
        MO("--newviewports",
           action="store_true", dest="newviewports", default=False,
           help="Use the new multiple viewport canvas implementation. This is an experimental feature only intended for developers.",
           ),
        MO("--testmodule",
           action="store", dest="testmodule", default=None,
           help="Run the docstring tests for module TESTMODULE. TESTMODULE is the name of the module, using . as path separator.",
           ),
        MO("--testcamera",
           action="store_true", dest="testcamera", default=False,
           help="Print camera settings whenever they change.",
           ),
        MO("--testexecutor",
           action="store_true", dest="executor", default=False,
           help="Test alternate executor: only for developers!",
           ),
        MO("--fastnurbs",
           action="store_true", dest="fastnurbs", default=False,
           help="Test C library nurbs drawing: only for developers!",
           ),
        MO("--listfiles",
           action="store_true", dest="listfiles", default=False,
           help="List the pyformex Python source files.",
           ),
        MO("--search",
           action="store_true", dest="search", default=False,
           help="Search the pyformex source for a specified pattern and exit. This can optionally be followed by -- followed by options for the grep command. Adding -a  will use the extended search path. The final argument is the pattern to search.",
           ),
        MO("--remove",
           action="store_true", dest="remove", default=False,
           help="Remove the pyFormex installation and exit. This option only works when pyFormex was installed from a tarball release using the supplied install procedure. If you install from a distribution package (e.g. Debian), you should use your distribution's package tools to remove pyFormex. If you run pyFormex directly from SVN sources, you should just remove the whole checked out source tree.",
           ),
        MO("--whereami",
           action="store_true", dest="whereami", default=False,
           help="Show where the pyformex package is installed and exit",
           ),
        MO("--detect",
           action="store_true", dest="detect", default=False,
           help="Show detected helper software and exit",
           ),
        ])
    pf.options, args = parser.parse_args(argv)
    pf.print_help = parser.print_help


    # Set debug level
    if pf.options.debug and not pf.options.debuglevel:
        pf.options.debuglevel = pf.debugLevel(pf.options.debug.split(','))

    # process options
    if pf.options.nodefaultconfig and not pf.options.config:
        print("\nInvalid options: --nodefaultconfig but no --config option\nDo pyformex --help for help on options.\n")
        sys.exit()


    pf.debug("Options: %s" % pf.options,pf.DEBUG.ALL)

    ########## Process special options which will not start pyFormex #######

    if pf.options.testmodule:
        for a in pf.options.testmodule.split(','):
            test_module(a)
        return

    if pf.options.remove:
        remove_pyFormex(pyformexdir,pf.bindir)
        return

    if pf.options.whereami: # or pf.options.detect :
        pf.options.debuglevel |= pf.DEBUG.INFO
            
    pf.debug("pyformex script started from %s" % pf.bindir,pf.DEBUG.INFO)
    pf.debug("I found pyFormex installed in %s " %  pyformexdir,pf.DEBUG.INFO)
    pf.debug("Current Python sys.path: %s" % sys.path,pf.DEBUG.INFO)

    if pf.options.detect:
        print("Detecting installed helper software")
        utils.checkExternal()
        print(utils.reportDetected())

    if pf.options.whereami or pf.options.detect :
        return

    ########### Read the config files  ####################

    # These values should not be changed
    pf.cfg.userprefs = os.path.join(pf.cfg.userconfdir,'pyformexrc')
    pf.cfg.autorun = os.path.join(pf.cfg.userconfdir,'startup.py')
    
    # Set the config files
    if pf.options.nodefaultconfig:
        sysprefs = []
        userprefs = []
    else:
        sysprefs = [ pf.cfg.siteprefs ]
        userprefs = [ pf.cfg.userprefs ]
        if os.path.exists(pf.cfg.localprefs):
            userprefs.append(pf.cfg.localprefs)

    if pf.options.config:
        userprefs.append(pf.options.config)

    if len(userprefs) == 0:
        # We should always have a place to store the user preferences
        userprefs = [ pf.cfg.userprefs ]

    pf.preffile = os.path.abspath(userprefs[-1]) # Settings will be saved here
   
    # Read all but the last as reference
    for f in filter(os.path.exists,sysprefs + userprefs[:-1]):
        pf.debug("Reading config file %s" % f,pf.DEBUG.CONFIG)
        pf.cfg.read(f)
     
    pf.refcfg = pf.cfg
    pf.debug("="*60,pf.DEBUG.CONFIG)
    pf.debug("RefConfig: %s" % pf.refcfg,pf.DEBUG.CONFIG)

    # Use the last as place to save preferences
    pf.prefcfg = Config(default=refLookup)
    if os.path.exists(pf.preffile):
        pf.debug("Reading config file %s" % pf.preffile,pf.DEBUG.CONFIG)
        pf.prefcfg.read(pf.preffile)
    pf.debug("="*60,pf.DEBUG.CONFIG)
    pf.debug("Config: %s" % pf.prefcfg,pf.DEBUG.CONFIG)

    # Fix incompatible changes in configuration
    apply_config_changes(pf.prefcfg)

    # Create an empty one for the session settings
    pf.cfg = Config(default=prefLookup)

    ####################################################################
    ## Post config initialization ##

    # process non-starting options dependent on config

    if pf.options.search or pf.options.listfiles:
        if len(args) > 0:
            opts = [ a for a in args if a.startswith('-') ]
            args = [ a for a in args if not a in opts ]
            if '-a' in opts:
                opts.remove('-a')
                extended = True
            else:
                extended = False
            if len(args) > 1:
                files = args[1:]
            else:
                files = utils.sourceFiles(relative=True,extended=extended)
            if pf.options.listfiles:
                print('\n'.join(files))
            else:
                cmd = "grep %s '%s' %s" % (' '.join(opts),args[0],''.join([" '%s'" % f for f in files]))
                #print cmd 
                os.system(cmd)
        return


    # process options that override the config
    if pf.options.redirect is not None:
        pf.cfg['gui/redirect'] = pf.options.redirect
    delattr(pf.options,'redirect') # avoid abuse
    #print "REDIRECT",pf.cfg['gui/redirect']

    ###################################################################

    
    # This should probably be changed to options overriding config
    # Set option from config if it was not explicitely given
    if pf.options.uselib is None:
        pf.options.uselib = pf.cfg['uselib']


    # Set default --nogui if first remaining argument is a pyformex script.
    if pf.options.gui is None:
        pf.options.gui = not (len(args) > 0 and utils.is_pyFormex(args[0]))

    if pf.options.gui:
        pf.options.interactive = True

    #  If we run from an SVN version, we should set the proper revision
    #  number and run the svnclean procedure.
    if pf.installtype=='S':
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
        ## m = re.match(".*//(?P<user>[^@]*)@svn\.savanna\.nongnu\.org.*",s)
        ## pf.svnuser = m.group('user')
        ## print pf.svnuser
    


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


    if pf.cfg['warnings/nice']:
        import warnings
        warnings.formatwarning = _format_warning

    # Make sure pf.PF is a Project
    from project import Project
    pf.PF = Project()

    utils.setSaneLocale()

    # Set application paths
    pf.debug("Loading AppDirs",pf.DEBUG.INFO)
    import apps
    apps.setAppDirs()

    # Start the GUI if needed
    # Importing the gui should be done after the config is set !!
    if pf.options.gui:
        from gui import guimain
        pf.debug("GUI version",pf.DEBUG.INFO)
        res = guimain.startGUI(args)
        if res != 0:
            print("Could not start the pyFormex GUI: %s" % res)
            return res # EXIT

    # Display the startup warnings and messages
    if startup_warnings:
        if pf.cfg['startup_warnings']:
            pf.warning(startup_warnings)
        else:
            print(startup_warnings)
    if startup_messages:
        print(startup_messages)

    if pf.options.debuglevel & pf.DEBUG.INFO:
        # NOTE: inside an if to avoid computing the report when not printed
        pf.debug(utils.reportDetected(),pf.DEBUG.INFO)
 
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
    #lib.init_libs(pf.options.uselib,pf.options.gui)

    # Prepend the autorun scripts
    ar = pf.cfg.get('autorun','')
    if ar :
        if type(ar) is str:
            ar = [ ar ]
        # expand tilde, as would bash
        ar = map(utils.tildeExpand,ar)
        args[0:0] = [ fn for fn in ar if os.path.exists(fn) ]

    # remaining args are interpreted as scripts and their parameters
    res = 0
    if args:
        pf.debug("Remaining args: %s" % args,pf.DEBUG.INFO)
        from script import processArgs
        res = processArgs(args)
        
        if res:
            if pf.options.gui:
                pf.message("There was an error while executing a script")
            else:
                return res # EXIT
                
    else:
        pf.debug("stdin is a tty: %s" % sys.stdin.isatty(),pf.DEBUG.INFO)
        # Play script from stdin
        # Can we check for interactive session: stdin connected to terminal?
        #from script import playScript
        #playScript(sys.stdin)
        

    # after processing all args, go into interactive mode
    if pf.options.gui and pf.app:
        res = guimain.runGUI()

    ## elif pf.options.interactive:
    ##     print("Enter your script and end with CTRL-D")
    ##     from script import playScript
    ##     playScript(sys.stdin)
        
    #Save the preferences that have changed
    savePreferences()

    # Exit
    return res


# End
