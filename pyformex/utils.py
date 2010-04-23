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
"""A collection of miscellaneous utility functions."""

import pyformex

import os,commands,re,sys
from config import formatDict
from distutils.version import LooseVersion as SaneVersion


# versions of detected modules/external commands
the_version = {
    'pyformex':pyformex.__version__,
    'python':sys.version.split()[0],
    }
the_external = {}

known_modules = [ 'python','numpy','pyopengl','pyqt4','pyqt4gl','calpy',
                  'gnuplot','gl2ps' ]

known_externals = {
    'Python': ('python --version','Python (\\S+)'),
    'ImageMagick': ('import -version','Version: ImageMagick (\S+)'),
    'admesh': ('admesh --version', 'ADMesh - version (\S+)'),
    'calpy': ('calpy --version','Calpy (\S+)'), 
    'tetgen': ('tetgen -h |fgrep Version','Version (\S+)'),
    'units': ('units --version','GNU Units version (\S+)'),
    'ffmpeg': ('ffmpeg -version','FFmpeg version (\\S+)'),
    'gts': ('gtsset -h','Usage(:) set'),
    'calix': ('calix --version','CALIX-(\S+)'),
    }

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
        ver = hasExternal(name)
    else:
        ver = hasModule(name)
    if SaneVersion(ver) > SaneVersion(version):
        return 1
    elif SaneVersion(ver) == SaneVersion(version):
        return 0
    else:
        return -1

    
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

    

def checkModule(name=None):
    """Check if the named Python module is available, and record its version.

    The version string is returned, empty if the module could not be loaded.
    The (name,version) pair is also inserted into the the_version dict.
    """
    if name is None:
        [ checkModule(n) for n in known_modules ]
        return
    
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
            version = str(gl2ps.GL2PS_VERSION)
    except:
        pass
    # make sure version is a string (e.g. gl2ps uses a float!)
    version = str(version)
    _congratulations(name,version,'module',fatal,quiet=True)
    the_version[name] = version
    return version



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
    _congratulations(name,version,'program',quiet=True)
    the_external[name] = version
    return version


def reportDetected():
    s = "%s (%s)\n\n" % (pyformex.Version,pyformex.__revision__)
    s += "Python version: %s\n" % sys.version
    s += "Operating system: %s\n\n" % sys.platform
    s += "Detected Python Modules:\n"
    for k,v in the_version.items():
        if not v:
            v = 'Not Found'
        s += "%s (%s)\n" % ( k,v)
    s += "\nDetected External Programs:\n"
    for k,v in the_external.items():
        #if not v:
        #    v = 'Not Found'
        s += "%s (%s)\n" % ( k,v)
    return s


    
def procInfo(title):
    print(title)
    print('module name: %s' %  __name__)
    print('parent process: %s' % os.getppid())
    print('process id: %s' % os.getpid())

def strNorm(s):
    """Normalize a string.

    Text normalization removes all '&' characters and converts it to lower case.
    """
    return str(s).replace('&','').lower()


def _congratulations(name,version,typ='module',fatal=False,quiet=True):
    """Report a detected module/program."""
    if version and not quiet:
        pyformex.message("Congratulations! You have %s (%s)" % (name,version))
    if not version:
        if not quiet or fatal:
            pyformex.message("ALAS! I could not find %s '%s' on your system" % (typ,name))
        if fatal:
            pyformex.message("Sorry, I'm out of here....")
            sys.exit()


def prefixFiles(prefix,files):
    """Prepend a prefix to a list of filenames."""
    return [ os.path.join(prefix,f) for f in files ]


def matchMany(regexps,target):
    """Return multiple regular expression matches of the same target string."""
    return [re.match(r,target) for r in regexps]


def matchCount(regexps,target):
    """Return the number of matches of target to  regexps."""
    return len(filter(None,matchMany(regexps,target)))
                  

def matchAny(regexps,target):
    """Check whether target matches any of the regular expressions."""
    return matchCount(regexps,target) > 0


def matchNone(regexps,target):
    """Check whether targes matches none of the regular expressions."""
    return matchCount(regexps,target) == 0


def matchAll(regexps,target):
    """Check whether targets matches all of the regular expressions."""
    return matchCount(regexps,target) == len(regexps)


def listTree(path,listdirs=True,topdown=True,sorted=False,excludedirs=[],excludefiles=[],includedirs=[],includefiles=[]):
    """List all files in path.

    If ``dirs==False``, directories are not listed.
    By default the tree is listed top down and entries in the same directory
    are unsorted.
    
    `exludedirs` and `excludefiles` are lists of regular expressions with
    dirnames, resp. filenames to exclude from the result.

    `includedirs` and `includefiles` can be given to include only the
    directories, resp. files matching any of those patterns.

    Note that 'excludedirs' and 'includedirs' force top down handling.
    """
    filelist = []
    if excludedirs or includedirs:
        topdown = True
    for root, dirs, files in os.walk(path, topdown=topdown):
        if sorted:
            dirs.sort()
            files.sort()
        if excludedirs:
            remove = [ d for d in dirs if matchAny(excludedirs,d) ]
            for d in remove:
                dirs.remove(d)
        if includedirs:
            remove = [ d for d in dirs if not matchAny(includedirs,d) ]
            for d in remove:
                dirs.remove(d)
        if listdirs and topdown:
            filelist.append(root)
        if excludefiles:
            files = [ f for f in files if matchNone(excludefiles,f) ]
        if includefiles:
            files = [ f for f in files if matchAny(includefiles,f) ]
        filelist.extend(prefixFiles(root,files))
        if listdirs and not topdown:
            filelist.append(root)
    return filelist


def removeTree(path,top=True):
    """Remove all files below path. If top==True, also path is removed."""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if top:
        os.rmdir(path)


###################### locale ###################

def setSaneLocale(localestring=''):
    """Set a sane local configuration for LC_NUMERIC.

    `locale` is the locale string to be set, e.g. 'en_US.UTF-8'

    This will change the ``LC_ALL`` setting to the specified string,
    and set the ``LC_NUMBERIC`` to 'C'.

    Changing the LC_NUMERIC setting is a very bad idea! It makes floating
    point values to be read or written with a comma instead of a the decimal
    point. Of course this makes input and output files completely incompatible.
    You will often not be able to process these files any further and
    create a lot of troubles for yourself and other people if you use an
    LC_NUMERIC setting different from the standard.

    Because we do not want to help you shoot yourself in the foot, this
    function always sets ``LC_NUMERIC`` back to a sane value and we
    call this function when pyFormex is starting up.
    """
    import locale
    locale.setlocale(locale.LC_ALL,localestring)
    locale.setlocale(locale.LC_NUMERIC, 'C')


###################### ReST conversion ###################

try:
    from docutils.core import publish_string
    def rst2html(text,writer='html'):
        return publish_string(text,writer_name=writer)
except ImportError:
    def rst2html(text,writer='html'):
        return text

###################### dos to unix conversion ###################

def dos2unix(infile,outfile=None):
    if outfile is None:
        cmd = "sed -i 's|\\r||' %s" % infile
    else:
        cmd = "sed -i 's|\\r||' %s > %s" % (infile,outfile)
    return runCommand(cmd)

def unix2dos(infile,outfile=None):
    if outfile is None:
        cmd = "sed -i 's|$|\\r|' %s" % infile
    else:
        cmd = "sed -i 's|$|\\r|' %s > %s" % (infile,outfile)
    return runCommand(cmd)

        


###################### image and file formats ###################

def all_image_extensions():
    """Return a list with all known image extensions."""
    imgfmt = []
    

file_description = {
    'all': 'All files (*)',
    'img': 'Images (*.png *.jpg *.eps *.gif)',
    'png': 'PNG images (*.png)',
    'icon': 'Icons (*.xpm)',
    'formex': 'Formex files (*.formex)',
    'gts': 'GTS files (*.gts)',
    'stl': 'STL files (*.stl)',
    'off': 'OFF files (*.off)',
    'smesh': 'Tetgen surface mesh files (*.smesh)',
    'neu': 'Gambit Neutral files (*.neu)',
    'surface': 'Any Surface file ( *.gts *.stl *.off *.smesh *.neu)',
    'postproc': 'Postproc scripts (*_post.py *.post)',
    'pyformex': 'pyFormex scripts (*.py *.pye)',
    'pyf': 'pyFormex projects (*.pyf)',
    'pgf': 'pyFormex geometry files (*.pgf *.formex)',
}

def fileDescription(ftype):
    """Return a description of the specified file type.

    The description of known types are listed in a dict file_description.
    If the type is unknown, the returned string has the form
    ``TYPE files (*.type)``
    """
    if type(ftype) is list:
        return map(fileDescription,ftype)
    ftype = ftype.lower()
    return file_description.get(ftype,"%s files (*.%s)" % (ftype.upper(),ftype))


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


def timeEval(s,glob=None):
    """Return the time needed for evaluating a string.

    s is a string with a valid Python instructions.
    The string is evaluated using Python's eval() and the difference
    in seconds between the current time before and after the evaluation
    is printed. The result of the evaluation is returned.

    This is a simple method to measure the time spent in some operation.
    It should not be used for microlevel instructions though, because
    the overhead of the time calls. Use Python's timeit module to measure
    microlevel execution time.
    """
    import time
    start = time.time()
    res = eval(s,glob)
    stop = time.time()
    print "Timed evaluation: %s seconds" % (stop-start)
    return res


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
#    if GD.GUI:
#        GD.GUI.setBusy(True)
    sta,out = commands.getstatusoutput(cmd)
#    if GD.GUI:
#        GD.GUI.setBusy(False)
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


def killProcesses(pids,signal):
    """Send the specified signal to the processes in list"""
    for pid in pids:
        try:
            os.kill(pid,signal)
        except:
            pyformex.debug("Error in killing of process '%s'" % pid)
            


def changeExt(fn,ext):
    """Change the extension of a file name.

    The extension is the minimal trailing part of the filename starting
    with a '.'. If the filename has no '.', the extension will be appended.
    If the given extension does not start with a dot, one is prepended.
    """
    if not ext.startswith('.'):
        ext = ".%s" % ext
    return os.path.splitext(fn)[0] + ext


def tildeExpand(fn):
    """Perform tilde expansion on a filename.

    Bash, the most used command shell in Linux, expands a '~' in arguments
    to the users home direction.
    This function can be used to do the same for strings that did not receive
    the bash tilde expansion, such as strings in the configuration file.
    """
    return fn.replace('~',os.environ['HOME'])
    

      
def isPyFormex(filename):
    """Checks whether a file is a pyFormex script.

    A script is considered to be a pyFormex script if its first line
    starts with '#!' and contains the substring 'pyformex'
    A file is considered to be a pyFormex script if its name ends in '.py'
    and the first line of the file contains the substring 'pyformex'.
    Typically, a pyFormex script starts with a line::

       #!/usr/bin/env pyformex
    """
    if filename.endswith(".pye"):
        return True

    ok = filename.endswith(".py")
    if ok:
        try:
            f = file(filename,'r')
            ok = f.readline().strip().find('pyformex') >= 0
            f.close()
        except IOError:
            ok = False
    return ok
    



class NameSequence(object):
    """A class for autogenerating sequences of names.

    The name is a string including a numeric part, which is incremented
    at each call of the 'next()' method.

    The constructor takes name template and a possible extension as arguments.
    If the name starts with a non-numeric part, it is taken as a constant
    part.
    If the name ends with a numeric part, the next generated names will
    be obtained by incrementing this part. If not, a string '-000' will
    be appended and names will be generated by incrementing this part.

    If an extension is given, it will be appended as is to the names.
    This makes it possible to put the numeric part anywhere inside the
    names.

    Examples:

    - ``NameSequence('hallo.98')`` generates the names
      hallo.98, hallo.99, hallo.100, ...
    - ``NameSequence('hallo','.png')`` generates the names
      hallo-000.png, hallo-001.png, ...
    - ``NameSequence('/home/user/hallo23','5.png')`` generates the names
      /home/user/hallo235.png, /home/user/hallo245.png, ...
    """
    
    def __init__(self,name,ext=''):
        """Create a new NameSequence from name,ext.

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
digits_string = re.compile('(\d*)(.*)$')

def splitEndDigits(s):
    """Split a string in any prefix and a numerical end sequence.

    A string like 'abc-0123' will be split in 'abc-' and '0123'.
    Any of both can be empty.
    """
    return string_digits.match(s).groups()


def splitStartDigits(s):
    """Split a string in a numerical sequence and any suffix.

    A string like '0123-abc' will be split in '0123' and '-abc'.
    Any of both can be empty.
    """
    return digits_string.match(s).groups()


def prefixDict(d,prefix=''):
    """Prefix all the keys of a dict with the given prefix.

    - `d`: a dict where all the keys are strings.
    - `prefix`: a string

    The return value is a dict with all the items of d, but where the
    keys have been prefixed with the given string.
    """
    return dict([ (prefix+k,v) for k,v in d.items() ])


def subDict(d,prefix=''):
    """Return a dict with the items whose key starts with prefix.

    - `d`: a dict where all the keys are strings.
    - `prefix`: a string
    
    The return value is a dict with all the items from d whose key starts
    with prefix. The keys in the returned dict will have the prefix
    stripped off.
    """
    return dict([ (k.replace(prefix,'',1),v) for k,v in d.items() if k.startswith(prefix)])

    
def stuur(x,xval,yval,exp=2.5):
    """Returns a (non)linear response on the input x.

    xval and yval should be lists of 3 values:
    ``[xmin,x0,xmax], [ymin,y0,ymax]``.
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
    import odict
    info = odict.ODict()
    if hasattr(item, '__name__'):
        info["NAME:    "] =  item.__name__
    if hasattr(item, '__class__'):
        info["CLASS:   "] = item.__class__.__name__
    info["ID:      "] = id(item)
    info["TYPE:    "] = type(item)
    info["VALUE:   "] = repr(item)
    info["CALLABLE:"] = callable(item)
    if hasattr(item, '__doc__'):
        doc = getattr(item, '__doc__')
        doc = doc.strip()   # Remove leading/trailing whitespace.
        firstline = doc.split('\n')[0]
        info["DOC:     "] = firstline
    for i in info.items():
        print("%s %s"% i) 


def deprecation(message):
    def decorator(func):
        def wrapper(*_args,**_kargs):
            import warnings
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*_args,**_kargs)
        return wrapper
    return decorator
    

def deprecated(replacement):
    def decorator(func):
        def wrapper(*_args,**_kargs):
            """This function is deprecated."""
            print("! Function '%s' is deprecated: use '%s.%s' instead" % (func.func_name,replacement.__module__,replacement.func_name))
            return replacement(*_args,**_kargs)
        return wrapper
    decorator.__doc__ = replacement.__doc__
    return decorator


def functionWasRenamed(replacement,text=None):
    def decorator(func):
        def wrapper(*_args,**_kargs):
            print("! Function '%s' is deprecated: use '%s' instead" % (func.func_name,replacement.func_name))
            return replacement(*_args,**_kargs)
        return wrapper
    decorator.__doc__ = replacement.__doc__
    return decorator


def functionBecameMethod(replacement):
    def decorator(func):
        def wrapper(object,*args,**kargs):
            print("! Function %s is deprecated: use method %s instead" % (func.func_name,replacement))
            repfunc = getattr(object,replacement)
            return repfunc(*args,**kargs)
        return wrapper
    return decorator

### End
