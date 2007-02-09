#!/usr/bin/env python
# $Id$
"""A collection of misc. utility functions."""

import globaldata as GD
import os,commands,re


known_externals = {
    'ImageMagick': ('import -version','Version: ImageMagick (\S+)'),
    'admesh': ('admesh --version', 'ADMesh - version (\S+)'),
    'tetgen': ('tetgen -v','Version (\S+)'), 
    }


def checkExternal(name,command=None,answer=None):
    """Check if the named external command is available on the system.

    name is the generic command name,
    command is the command as it will be executed to check its operation,
    answer is a regular expression to match positive answers from the command.
    answer should contain at least one group. In case of a match, the
    contents of the match will be stored in the GD.external dict
    with name as the key. If the result does not match the specified answer,
    an empty value is inserted.

    Usually, command will contain an option to display the version, and
    the answer re contains a group to select the version string from
    the result.

    As a convenience, we provide a list of predeclared external commands,
    that can be checked by their name alone.
    """
    if command is None or answer is None:
        cmd,ans = known_externals.get(name,(name,'(.+)\n'))
        if command is None:
            command = cmd
        if answer is None:
            answer = ans
    m = re.match(answer,commands.getoutput(command))
    if m:
        value = m.group(1)
        GD.message("Congratulations! You have %s (%s)" % (name,value))
    else:
        value = ''
    GD.external[name] = value
    return value

    
def hasExternal(name):
    """Test if we have the external command 'name' available.

    Returns a nonzero string if the command is available,
    or an empty string if it is not.

    The external command is only checked on the first call.
    The result is remembered in the GD.external dict.
    """
    if GD.external.has_key(name):
        return GD.external[name]
    else:
        return checkExternal(name)


file_description = {
    'stl/off': 'STL or OFF files (*.stl *.off *.neu *.smesh *.gts)',
    'stl': 'STL files (*.stl)',
    'off': 'OFF files (*.off)',
    'gts': 'GTS files (*.gts)',
    'neu': 'Gambit Neutral files (*.neu)',
    'smesh': 'Tetgen surface mesh files (*.smesh)',
}

def fileDescription(type):
    """Return a description of the specified file type.

    The description of known types are liste in a dict file_description.
    If the type is unknown, the returned string has the form
    'TYPE files (*.type)'
    """
    return file_description.get(type,"%s files (*.%s)" % (type.upper(),type))
    

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


def runCommand(cmd,RaiseError=True):
    """Run a command and raise error if exited with error."""
    GD.message("Running command: %s" % cmd)
    sta,out = commands.getstatusoutput(cmd)
    if sta != 0 and RaiseError:
        raise RuntimeError, "Error while executing command:\n  %s" % cmd
    return sta,out


def spawn(cmd):
    """Spawn a child process."""
    cmd = cmd.split()
    pid = os.spawnvp(os.P_NOWAIT,cmd[0],cmd)
    GD.debug("Spawned child process %s for command '%s'" % (pid,cmd))
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

class FilenameSequence(object):
    """A class for autogenerating sequences of file names.

    The file name includes a numeric part, whose number is incremented
    at each call of the 'next()' method.
    """
    
    def __init__(self,filename,ext=''):
        """Create a new FilenameSequence from name,ext.

        The filename can include a path or not.
        If the filename ends with a numeric part, the next generated
        filenames will be obtained by incrementing this part.
        If not, a string '-000' will be appended and filenames will
        be generated by incrementing this part.

        If an extension is given, it will be appended as is to the filename.
        This makes it possible to put the numeric part anywhere inside the
        filename.

        Examples:
            FilenameSequence('hallo','.png') will generate names
                hallo-000.png, hallo-001.png, ...
            FilenameSequence('dir/hallo23','5.png') will generate names
                dir/hallo235.png, hallo245.png, ...
        """
        name,number = splitEndDigits(filename)
        if len(number) > 0:
            self.nr = int(number)
            format = "%%0%dd" % len(number)
        else:
            self.nr = 0
            format = "-%03d"
        self.name = name+format+ext

    def next(self):
        """Return the next filename in the sequence"""
        fn = self.name % self.nr
        self.nr += 1
        return fn


def imageFormatFromExt(ext):
    """Determine the image format from an extension.

    The extension may or may not have an initial dot and may be in upper or
    lower case. The format is equal to the extension characters in lower case.
    If the supplied extension is empty, the default format 'png' is returned.
    """
    if len(ext) > 0:
        if ext[0] == '.':
            ext = ext[1:]
        fmt = ext.lower()
    else:
        fmt = 'png'
    return fmt


def splitEndDigits(s):
    """Split a string in any prefix and a numerical end sequence.

    A string like 'abc-0123' will be split in 'abc-' and '0123'.
    Any of both can be empty.
    """
    i = len(s)
    if i == 0:
        return ( '', '' )
    i -= 1
    while s[i].isdigit() and i > 0:
        i -= 1
    if not s[i].isdigit():
        i += 1
    return ( s[:i], s[i:] )


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


def deprecated2(func, name=None):
    if name is None:
        name = func.__name__
    def wrapped(*args, **kargs):
        print "Calling", name, args, kargs
        result = func(*args, **kargs)
        print "Called", name, args, kargs, "returned", repr(result)
        return result
    wrapped.__doc__ = func.__doc__
    return wrapped

def formatDict(dic):
    """Format a dict in Python source representation.

    Each (key,value) pair is formatted on a line : key = value.
    """
    s = ""
    if isinstance(dic,dict):
        for k,v in dic.iteritems():
            if type(v) == str:
                s += '%s = "%s"\n' % (k,v)
            else:
                s += '%s = %s\n' % (k,v)
    return s

