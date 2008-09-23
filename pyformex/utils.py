# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""A collection of misc. utility functions."""

import pyformex

import os,commands,re,sys
from config import formatDict
from numpy import unique1d,union1d,setdiff1d
from distutils.version import LooseVersion as SaneVersion


# versions of detected modules/external commands
the_version = {'pyformex':pyformex.__version__}
the_external = {}

def congratulations(name,version,typ='module',fatal=False):
    """Report a detected module/program."""
    if version and pyformex.options.debug:
        pyformex.message("Congratulations! You have %s (%s)" % (name,version))
    if not version:
        if pyformex.options.debug or fatal:
            pyformex.message("ALAS! I could not find %s '%s' on your system" % (typ,name))
        if fatal:
            pyformex.message("Sorry, I'm out of here....")
            sys.exit()

def checkVersion(name,version,external=False):
    """Checks a version of a program/module.

    name is either a module or an external program whose availability has
    been registered.
    Default is to treat name as a module. Add external=True for a program.

    Return value is -1, 0 or 1, depending on a version found that is
    <, == or > than the requested values.
    This should normally understand version numbers in the format 2.10.1
    """
    if external:
        ver = the_external.get(name,'0')
    else:
        ver = the_version.get(name,'0')
    if SaneVersion(ver) > SaneVersion(version):
        return 1
    elif SaneVersion(ver) == SaneVersion(version):
        return 0
    else:
        return -1

            
def checkModule(name):
    """Check if the named Python module is available, and record its version.

    The version string is returned, empty if the module could not be loaded.
    The (name,version) pair is also inserted into the the_version dict.
    """
    version = ''
    fatal = False
    try:
        if name == 'numpy':
            fatal = True
            import numpy
            version =  numpy.__version__
        elif name == 'pyopengl':
            fatal = pyformex.options.gui
            import OpenGL
            version =  OpenGL.__version__
        elif name == 'pyqt4':
            fatal = pyformex.options.gui
            import PyQt4.QtCore
            version = PyQt4.QtCore.QT_VERSION_STR
        elif name == 'pyqt4gl':
            fatal = pyformex.options.gui
            import PyQt4.QtOpenGL
            import PyQt4.QtCore
            version = PyQt4.QtCore.QT_VERSION_STR
        elif name == 'calpy':
            import calpy
            version = calpy.__version__
        elif name == 'gnuplot':
            import Gnuplot
            version = Gnuplot.__version__
        elif name == 'gl2ps':
            import gl2ps
            version = gl2ps.GL2PS_VERSION
    except:
        pass
    congratulations(name,version,'module',fatal)
    the_version[name] = version
    return version

    
def hasModule(name,check=False):
    """Test if we have the named module available.

    Returns a nonzero (version) string if the module is available,
    or an empty string if it is not.

    By default, the module is only checked on the first call. 
    The result is remembered in the the_version dict.
    The optional argument check==True forces a new detection.
    """
    if the_version.has_key(name) and not check:
        return the_version[name]
    else:
        return checkModule(name)

            
# tetgen -v no longer works in 1.4.2: use -h !!
known_externals = {
    'ImageMagick': ('import -version','Version: ImageMagick (\S+)'),
    'admesh': ('admesh --version', 'ADMesh - version (\S+)'),
    'calpy': ('calpy --version','Calpy (\S+)'), 
    'tetgen': ('tetgen -h |fgrep Version','Version (\S+)'),
    'units': ('units --version','GNU Units version (\S+)'),
    'ffmpeg': ('ffmpeg -version','FFmpeg version (\\S+)'),
    }


def checkExternal(name=None,command=None,answer=None):
    """Check if the named external command is available on the system.

    name is the generic command name,
    command is the command as it will be executed to check its operation,
    answer is a regular expression to match positive answers from the command.
    answer should contain at least one group. In case of a match, the
    contents of the match will be stored in the the_external dict
    with name as the key. If the result does not match the specified answer,
    an empty value is inserted.

    Usually, command will contain an option to display the version, and
    the answer re contains a group to select the version string from
    the result.

    As a convenience, we provide a list of predeclared external commands,
    that can be checked by their name alone.
    If no name is given, all commands in that list are checked, and no
    value is returned.
    """
    if name is None:
        [ checkExternal(n) for n in known_externals.keys() ]
        return
    
    if command is None or answer is None:
        cmd,ans = known_externals.get(name,(name,'(.+)\n'))
        if command is None:
            command = cmd
        if answer is None:
            answer = ans

    m = re.match(answer,commands.getoutput(command))
    if m:
        version = m.group(1)
    else:
        version = ''
    congratulations(name,version,'program')
    the_external[name] = version
    return version


def hasExternal(name):
    """Test if we have the external command 'name' available.

    Returns a nonzero string if the command is available,
    or an empty string if it is not.

    The external command is only checked on the first call.
    The result is remembered in the the_external dict.
    """
    if the_external.has_key(name):
        return the_external[name]
    else:
        return checkExternal(name)


def printDetected():
    #detectAll()
    sta,out = runCommand('cd %s && svnversion' % pyformex.cfg['pyformexdir'],quiet=True)
    if sta == 0 and not out.startswith('exported'):
        pyformex.__revision__ = "$Rev: %s $" % out.strip()
    print "%s (%s)\n" % (pyformex.Version,pyformex.__revision__)
    print "Detected Python Modules:"
    for k,v in the_version.items():
        if v:
            print "%s (%s)" % ( k,v)
    print "\nDetected External Programs:"
    for k,v in the_external.items():
        if v:
            print "%s (%s)" % ( k,v)


def removeTree(path,top=True):
    """Remove all files below path. If top==True, also path is removed."""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if top:
        os.rmdir(path)


###################### image and file formats ###################

def all_image_extensions():
    """Return a list with all known image extensions."""
    imgfmt = []
    

file_description = {
    'all': 'All files (*)',
    'img': 'Images (*.png *.jpg *.eps)',
    'png': 'PNG images (*.png)',
    'icon': 'Icons (*.xpm)',
    'formex': 'Formex files (*.formex)',
    'gts': 'GTS files (*.gts)',
    'stl': 'STL files (*.stl)',
    'off': 'OFF files (*.off)',
    'smesh': 'Tetgen surface mesh files (*.smesh)',
    'neu': 'Gambit Neutral files (*.neu)',
    'surface': 'Any Surface file ( *.gts *.stl *.off *.smesh *.neu)',
    'postproc': 'Postproc scripts (*_post.py *.post)'
}

def fileDescription(type):
    """Return a description of the specified file type.

    The description of known types are listed in a dict file_description.
    If the type is unknown, the returned string has the form
    'TYPE files (*.type)'
    """
    return file_description.get(type,"%s files (*.%s)" % (type.upper(),type))


def findIcon(name):
    """Return the file name for an icon with given name.

    If no icon file is found, returns the question mark icon.
    """
    fname = os.path.join(pyformex.cfg['icondir'],name) + pyformex.cfg['gui/icontype']
    if os.path.exists(fname):
        return fname
    return os.path.join(pyformex.cfg['icondir'],'question') + pyformex.cfg['gui/icontype']
                                                               

def projectName(fn):
    """Derive a project name from a file name.

    The project name is the basename f the file without the extension.
    """
    return os.path.splitext(os.path.basename(fn))[0]


def splitme(s):
    return s[::2],s[1::2]


def mergeme(s1,s2):
    return ''.join([a+b for a,b in zip(s1,s2)])


def mtime(fn):
    """Return the (UNIX) time of last change of file fn."""
    return os.stat(fn).st_mtime


def countLines(fn):
    """Return the number of lines in a text file."""
    sta,out = runCommand("wc %s" % fn)
    if sta == 0:
        return int(out.split()[0])
    else:
        return 0


def runCommand(cmd,RaiseError=True,quiet=False):
    """Run a command and raise error if exited with error."""
    if not quiet:
        pyformex.message("Running command: %s" % cmd)
#    if GD.gui:
#        GD.gui.setBusy(True)
    sta,out = commands.getstatusoutput(cmd)
#    if GD.gui:
#        GD.gui.setBusy(False)
    if sta != 0:
        pyformex.message(out)
        if RaiseError:
            raise RuntimeError, "Error while executing command:\n  %s" % cmd
    return sta,out


def spawn(cmd):
    """Spawn a child process."""
    cmd = cmd.split()
    pid = os.spawnvp(os.P_NOWAIT,cmd[0],cmd)
    pyformex.debug("Spawned child process %s for command '%s'" % (pid,cmd))
    return pid


def changeExt(fn,ext):
    """Change the extension of a file name.

    The extension is the minimal trailing part of the filename starting
    with a '.'. If the filename has no '.', the extension will be appended.
    If the given extension does not start with a dot, one is prepended.
    """
    if not ext.startswith('.'):
        ext = ".%s" % ext
    return os.path.splitext(fn)[0] + ext

      
def isPyFormex(filename):
    """Checks whether a file is a pyFormex script.

    A script is considered to be a pyFormex script if its first line
    starts with '#!' and contains the substring 'pyformex'
    A file is considered to be a pyFormex script if its name ends in '.py'
    and the first line of the file contains the substring 'pyformex'.
    Typically, a pyFormex script starts with a line:
      #!/usr/bin/env pyformex
    """
    ok = filename.endswith(".py")
    if ok:
        try:
            f = file(filename,'r')
            ok = f.readline().strip().find('pyformex') >= 0
            f.close()
        except IOError:
            ok = False
    return ok


def sortOnLength(items):
    """Sort a list of lists according to length of the sublists.

    items is a list of items each having the len() method.
    The items are put in separate lists according to their length.

    The return value is a dict where the keys are item lengths and
    the values are lists of items with this length.
    """
    res = {}
    for item in items:
        li = len(item)
        if li in res.keys():
            res[li].append(item)
        else:
            res[li] = [ item ]
    return res


class NameSequence(object):
    """A class for autogenerating sequences of names.

    The name includes a numeric part, whose number is incremented
    at each call of the 'next()' method.
    """
    
    def __init__(self,name,ext=''):
        """Create a new NameSequence from name,ext.

        If the name starts with a non-numeric part, it is taken as a constant
        part.
        If the name ends with a numeric part, the next generated names will
        be obtained by incrementing this part.
        If not, a string '-000' will be appended and names will be generated
        by incrementing this part.

        If an extension is given, it will be appended as is to the names.
        This makes it possible to put the numeric part anywhere inside the
        names.

        Examples:
            NameSequence('hallo.98') will generate names
                hallo.98, hallo.99, hallo.100, ...
            NameSequence('hallo','.png') will generate names
                hallo-000.png, hallo-001.png, ...
            NameSequence('/home/user/hallo23','5.png') will generate names
                /home/user/hallo235.png, /home/user/hallo245.png, ...
        """
        base,number = splitEndDigits(name)
        if len(number) > 0:
            self.nr = int(number)
            format = "%%0%dd" % len(number)
        else:
            self.nr = 0
            format = "-%03d"
        self.name = base+format+ext

    def next(self):
        """Return the next name in the sequence"""
        fn = self.name % self.nr
        self.nr += 1
        return fn

    def peek(self):
        """Return the next name in the sequence without incrementing."""
        return self.name % self.nr

    def glob(self):
        """Return a UNIX glob pattern for the generated names.

        A NameSequence is often used as a generator for file names.
        The glob() method returns a pattern that can be used in a
        UNIX-like shell command to select all the generated file names.
        """
        i = self.name.find('%')
        j = self.name.find('d',i)
        return self.name[:i]+'*'+self.name[j+1:]


string_digits = re.compile('(.*?)(\d*)$')

def splitEndDigits(s):
    """Split a string in any prefix and a numerical end sequence.

    A string like 'abc-0123' will be split in 'abc-' and '0123'.
    Any of both can be empty.
    """
    return string_digits.match(s).groups()
    
    
def stuur(x,xval,yval,exp=2.5):
    """Returns a (non)linear response on the input x.

    xval and yval should be lists of 3 values:
      [xmin,x0,xmax], [ymin,y0,ymax].
    Together with the exponent exp, they define the response curve
    as function of x. With an exponent > 0, the variation will be
    slow in the neighbourhood of (x0,y0).
    For values x < xmin or x > xmax, the limit value ymin or ymax
    is returned.
    """
    xmin,x0,xmax = xval
    ymin,y0,ymax = yval 
    if x < xmin:
        return ymin
    elif x < x0:
        xr = float(x-x0) / (xmin-x0)
        return y0 + (ymin-y0) * xr**exp
    elif x < xmax:
        xr = float(x-x0) / (xmax-x0)
        return y0 + (ymax-y0) * xr**exp
    else:
        return ymax



def interrogate(item):
    """Print useful information about item."""
    if hasattr(item, '__name__'):
        print "NAME:    ", item.__name__
    if hasattr(item, '__class__'):
        print "CLASS:   ", item.__class__.__name__
    print "ID:      ", id(item)
    print "TYPE:    ", type(item)
    print "VALUE:   ", repr(item)
    print "CALLABLE:",
    if callable(item):
        print "Yes"
    else:
        print "No"
    if hasattr(item, '__doc__'):
        doc = getattr(item, '__doc__')
        doc = doc.strip()   # Remove leading/trailing whitespace.
        firstline = doc.split('\n')[0]
        print "DOC:     ", firstline


def deprecated(replacement):
    def decorator(func):
        def wrapper(*_args,**_kargs):
            print "Function %s is deprecated: use %s instead" % (func.func_name,replacement.func_name)
            return replacement(*_args,**_kargs)
        return wrapper
    decorator.__doc__ = replacement.__doc__
    return decorator


### End
