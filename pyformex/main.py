#!/usr/bin/python
# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
import pyformex as pf

import sys,os
startup_warnings = ''
startup_messages = ''

pyformexdir = sys.path[0]
pf.svnversion = os.path.exists(os.path.join(pyformexdir,'.svn'))
if pf.svnversion:

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
        print "Rebuilding pyFormex libraries, please wait"
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

# intended Python version
minimal_version = '2.5'
target_version = '2.7'
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
Your Python version is %s, but pyFormex has only been tested with Python <= %s. We expect pyFormex to run correctly with your Python version, but if you encounter problems, please contact the developers at http://pyformex.org.
""" % (found_version,target_version,)



from config import Config


# Remove unwanted warnings 
# We have moved this to the config file 
#utils.filterWarning('.*return_index.*','numpy')

###########################  main  ################################

def filterWarnings():
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
        pf.debug("!There is no key '%s' in the reference config!"%key,pf.DEBUG.WARNING)
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


def setRevision():
    sta,out = utils.runCommand('cd %s && svnversion' % pf.cfg['pyformexdir'],quiet=True)
    if sta == 0 and not out.startswith('exported'):
        pf.__revision__ = out.strip()


def remove_pyFormex(pyformexdir,scriptdir):
    """Remove the pyFormex installation."""
    print("""
BEWARE!
This procedure will remove the complete pyFormex installation!
You should only use this on a pyFormex installed with 'python setup.py install'.
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
           action="store", dest="debug", default='',
           help="display debugging info to sys.stdout",
           ),
        MO("--debuglevel",
           action="store", dest="debuglevel", type="int", default=0,
           help="display debugging info to sys.stdout. The value is an int with the bits of the requested debug levels set. A value of -1 switches on all debug info. If this option is used, it overrides the --debug option.",
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
        ## MO("--test",
        ##    action="store_true", dest="test", default=False,
        ##    help="testing mode: only for developers!",
        ##    ),
        MO("--testexecutor",
           action="store_true", dest="executor", default=False,
           help="test alternate executor: only for developers!",
           ),
        ## MO("--olddraw",
        ##    action="store_true", dest="olddraw", default=False,
        ##    help="use the old (slower) drawing function: use only when the new function gives problems",
        ##    ),
        MO("--fastnurbs",
           action="store_true", dest="fastnurbs", default=False,
           help="test C library nurbs drawing: only for developers!",
           ),
        MO("--listfiles",
           action="store_true", dest="listfiles", default=False,
           help="list the pyformex Python source files.",
           ),
        MO("--search",
           action="store_true", dest="search", default=False,
           help="search the pyformex source for a specified pattern and exit. This can optionally be followed by -- followed by options for the grep command. The final argument is the pattern to search.",
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
    pf.options, args = parser.parse_args(argv)
    pf.print_help = parser.print_help


    # process options
    if pf.options.nodefaultconfig and not pf.options.config:
        print("\nInvalid options: --nodefaultconfig but no --config option\nDo pyformex --help for help on options.\n")
        sys.exit()

    pf.debug("Options: %s" % pf.options,pf.DEBUG.OPTION)


    ########## Process special options which will not start pyFormex #######


    if pf.options.listfiles:
        print '\n'.join(utils.pyformexFiles(relative=True))
        return

    if pf.options.search:
        if len(args) > 0:
            #from script import grepSource
            #print grepSource(args[-1],' '.join(args[:-1]),quiet=True)
            options = [ a for a in args if a.startswith('-') ]
            args = [ a for a in args if not a in options ]
            if len(args) > 1:
                files = args[1:]
            else:
                files = utils.pyformexFiles(relative=True)
            cmd = "grep %s '%s' %s" % (' '.join(options),args[0],' '.join(files))
            os.system(cmd)
        return

    if pf.options.testmodule:
        for a in pf.options.testmodule.split(','):
            test_module(a)
        return

    if pf.options.remove:
        remove_pyFormex(pyformexdir,pf.scriptdir)
        return

    if pf.options.whereami or pf.options.detect :
        pf.options.debuglevel |= pf.DEBUG.INFO
            
    pf.debug("Script started from %s" % pf.scriptdir,pf.DEBUG.INFO)
    pf.debug("I found pyFormex in %s " %  pyformexdir,pf.DEBUG.INFO)
    pf.debug("Current Python sys.path: %s" % sys.path,pf.DEBUG.INFO)

    if pf.options.detect:
        print("Detecting all installed helper software")
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
        pf.debug("Reading config file %s" % f)
        pf.cfg.read(f)
     
    pf.refcfg = pf.cfg
    pf.debug("="*60)
    pf.debug("RefConfig: %s" % pf.refcfg)

    # Use the last as place to save preferences
    pf.prefcfg = Config(default=refLookup)
    if os.path.exists(pf.preffile):
        pf.debug("Reading config file %s" % pf.preffile)
        pf.prefcfg.read(pf.preffile)
    pf.debug("="*60)
    pf.debug("Config: %s" % pf.prefcfg)

    # Fix incompatible changes in configuration
    apply_config_changes(pf.prefcfg)

    # Create an empty one for the session settings
    pf.cfg = Config(default=prefLookup)

    # This should probably be changed to options overriding config
    # Set option from config if it was not explicitely given
    if pf.options.uselib is None:
        pf.options.uselib = pf.cfg['uselib']


    # Set default --nogui if first remaining argument is a pyformex script.
    if pf.options.gui is None:
        pf.options.gui = not (len(args) > 0 and utils.is_pyFormex(args[0]))

    if pf.options.gui:
        pf.options.interactive = True

    # Set Revision and run svnclean if we run from an SVN version
    if pf.svnversion:
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
        ## m = re.match(".*//(?P<user>[^@]*)@svn\.savanna\.nongnu\.org.*",s)
        ## pf.svnuser = m.group('user')
        ## print pf.svnuser

        # Add in subversion specific config
        devhowto = os.path.join(pyformexdir,'..','HOWTO-dev.rst')
        builddoc = os.path.join(pyformexdir,"doc","build-local-docs.rst")
        pf.refcfg.help['developer'][0:0] = [('Developer HOWTO',devhowto),('&Build local documentation',builddoc)]
        #print pf.refcfg.help['developer']
    


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

    # Start the GUI if needed
    # Importing the gui should be done after the config is set !!
    if pf.options.gui:
        from gui import guimain
        pf.debug("GUI version")
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

    if pf.options.debug: # Avoid computing the report if not printed
        pf.debug(utils.reportDetected())
 
    #print(pf.cfg.keys())
    #print(pf.refcfg.keys())
      
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
        pf.debug("Remaining args: %s" % args)
        from script import processArgs
        res = processArgs(args)
        
        if res:
            if pf.options.gui:
                pf.message("There was an error while executing a script")
            else:
                return res # EXIT
                
    else:
        pf.debug("stdin is a tty: %s" % sys.stdin.isatty())
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
