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
"""A collection of miscellaneous utility functions.

"""
from __future__ import print_function

import pyformex as pf

import os,re,sys,tempfile
from config import formatDict
from distutils.version import LooseVersion as SaneVersion

# Some regular expressions
digits = re.compile(r'(\d+)')

# versions of detected modules
the_version = {
    'pyformex':pf.__version__,
    'python':sys.version.split()[0],
    }
# versions of detected external commands
the_external = {}

known_externals = {
    'Python': ('python --version','Python (\\S+)'),
    'ImageMagick': ('import -version','Version: ImageMagick (\S+)'),
    'admesh': ('admesh --version', 'ADMesh - version (\S+)'),
    'calpy': ('calpy --version','Calpy (\S+)'), 
    'tetgen': ('tetgen -h |fgrep Version','Version (\S+)'),
    'units': ('units --version','GNU Units version (\S+)'),
    'ffmpeg': ('ffmpeg -version','FFmpeg version (\S+)'),
    'gts': ('gtsset -h','Usage(:) '),
    'calix': ('calix --version','CALIX-(\S+)'),
    'dxfparser': ('pyformex-dxfparser --version','dxfparser (\S+)'),
    'postabq': ('pyformex-postabq -V','postabq (\S+).*'),
    'vmtk': ('vmtk --help','Usage: 	vmtk(\S+).*'),
    'recordmydesktop': ('recordmydesktop --version','recordMyDesktop v(\S+)'),
    }

def checkVersion(name,version,external=False):
    """Checks a version of a program/module.

    name is either a module or an external program whose availability has
    been registered.
    Default is to treat name as a module. Add external=True for a program.

    Return value is -1, 0 or 1, depending on a version found that is
    <, == or > than the requested values.
    This should normally understand version numbers in the format 2.10.1
    Returns -2 if no version found.
    """
    if external:
        ver = hasExternal(name)
    else:
        ver = hasModule(name)
    if not ver:
        return -2
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
    if name in the_version and not check:
        return the_version[name]
    else:
        return checkModule(name)


def hasExternal(name,force=False):
    """Test if we have the external command 'name' available.

    Returns a nonzero string if the command is available,
    or an empty string if it is not.

    The external command is only checked on the first call.
    The result is remembered in the the_external dict.
    """
    if name in the_external and not force:
        return the_external[name]
    else:
        return checkExternal(name)


def requireModule(name):
    """Ensure that the named Python module is available.

    If the module is not available, an error is raised.
    """
    if not hasModule(name):
        errmsg = known_modules[name][0]
        if errmsg:
            pf.error(errmsg)
        
        

# Python modules we know how to use
# Do not include pyformex or python here: they are predefined
# and could be erased by the detection
# The value is a chain of attributes to get the version
# If empty, the attribute is supposed to be '__version__'
known_modules = {
    'numpy'     : (),
    'pyopengl'  : ('OpenGL',),
    'docutils'  : (),
    'calpy'     : (),
    'pyqt4'     : ('PyQt4','','QtCore','QT_VERSION_STR'),
    'pyqt4gl'   : ('PyQt4','','QtOpenGL'),
    'matplotlib': (),
    'gnuplot'   : ('Gnuplot'),
    'gl2ps'     : ('','','GL2PS_VERSION'),
    'vtk'       : ('','','VTK_VERSION'),
     }
    

def checkAllModules():
    """Check the existing of all known modules.

    """
    [ checkModule(n,quiet=True) for n in known_modules ]
    return 
    

def checkModule(name,ver=(),fatal=False,quiet=False):
    """Check if the named Python module is available, and record its version.

    ver is a tuple of:

    - modname: name of the module to test import
    - vername: name of the module holding the version string
    - more fields are consecutive attributes leading to the version string

    The obtained version string is returned, empty if the module could not
    be loaded.
    The (name,version) pair is also inserted into the the_version dict.

    If fatal=True, pyFormex will abort if the module can not be loaded.
    """
    if len(ver) == 0 and name in known_modules:
        ver = known_modules[name]

    modname = name
    if len(ver) > 0  and len(ver[0]) > 0:
        modname = ver[0]

    try:
        #print(modname)
        m = __import__(modname)
        #print(m)
        if len(ver) > 1 and len(ver[1]) > 0:
            modname = ver[1]
            m = __import__(modname)
            #print(m)
        ver = ver[2:]
        if len(ver) == 0:
            ver = ('__version__',)
        for a in ver:
            m = getattr(m,a)
            #print(m)

    except:
        # failure: unexisting or unregistered modules
        m = ''

    #print("Module %s: Version %s" % (name,m))
   
    # make sure version is a string (e.g. gl2ps uses a float!)
    m = str(m)
    _congratulations(name,m,'module',fatal,quiet=quiet)
    #if version:
    the_version[name] = m
    return m


def checkExternal(name=None,command=None,answer=None,quiet=False):
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
        [ checkExternal(n,quiet=True) for n in known_externals.keys() ]
        return
    
    if command is None or answer is None:
        cmd,ans = known_externals.get(name,(name,'(.+)\n'))
        if command is None:
            command = cmd
        if answer is None:
            answer = ans

    out = system2(command)[1]
    m = re.match(answer,out)
    if m:
        version = m.group(1)
    else:
        version = ''
    _congratulations(name,version,'program',quiet=quiet)
    #if version:
    the_external[name] = version
    return version


def _congratulations(name,version,typ='module',fatal=False,quiet=False,severity=2):
    """Report a detected module/program."""
    if version:
        if not quiet:
            pf.debug("Congratulations! You have %s (%s)" % (name,version),pf.DEBUG.DETECT)
    else:
        if not quiet or fatal:
            pf.debug("ALAS! I could not find %s '%s' on your system" % (typ,name),pf.DEBUG.DETECT)
        if fatal:
            pf.error("Sorry, I'm getting out of here....")
            sys.exit()


def FullVersion():
    return "%s (Rev. %s)" % (pf.Version,pf.__revision__)

def Libraries():
    from lib import accelerated
    acc = [ m.__name__ for m in accelerated ]
    return ', '.join(acc)

def reportDetected():
    notfound = '** Not Found **'
    s = "%s\n" % FullVersion()
    s += "\nInstall type: %s\n" % pf.installtype
    s += "\npyFormex C libraries: %s\n" % Libraries()
    s += "\nPython version: %s\n" % sys.version
    s += "\nOperating system: %s\n" % sys.platform
    s += "\nDetected Python Modules:\n"
    for k,v in the_version.iteritems():
        if not v:
            v = notfound
        s += "%s (%s)\n" % ( k,v)
    s += "\nDetected External Programs:\n"
    for k,v in the_external.iteritems():
        if not v:
            v = notfound
        s += "%s (%s)\n" % ( k,v)
    return s

    
def procInfo(title):
    print(title)
    print('module name: %s' %  __name__)
    print('parent process: %s' % os.getppid())
    print('process id: %s' % os.getpid())


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


def listTree(path,listdirs=True,topdown=True,sorted=False,excludedirs=[],excludefiles=[],includedirs=[],includefiles=[],symlinks=True):
    """List all files in path.

    If ``listdirs==False``, directories are not listed.
    By default the tree is listed top down and entries in the same directory
    are unsorted.
    
    `exludedirs` and `excludefiles` are lists of regular expressions with
    dirnames, resp. filenames to exclude from the result.

    `includedirs` and `includefiles` can be given to include only the
    directories, resp. files matching any of those patterns.

    Note that 'excludedirs' and 'includedirs' force top down handling.

    If `symlinks` is set False, symbolic links are removed from the list.
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
    if not symlinks:
        filelist = [ f for f in filelist if not os.path.islink(f) ]
    return filelist


def removeFile(filename):
    """Remove a file, ignoring error when it does not exist."""
    if os.path.exists(filename):
        os.remove(filename)


def removeTree(path,top=True):
    """Remove all files below path. If top==True, also path is removed."""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if top:
        os.rmdir(path)


def sourceFiles(relative=False,symlinks=True,extended=False):
    """Return a list of the pyFormex source .py files.

    - `symlinks`: if False, files that are symbolic links are retained in the
      list. The default is to remove them.
    - `extended`: if True, the .py files in all the paths in the configured
      appdirs and scriptdirs are also added.
    """
    path = pf.cfg['pyformexdir']
    if relative:
        path = os.path.relpath(path)
    files = listTree(path,listdirs=False,sorted=True,includedirs=['gui','plugins','apps','examples','lib'],includefiles=['.*\.py$'],symlinks=symlinks)
    if extended:
        searchdirs = [ i[1] for i in pf.cfg['appdirs'] + pf.cfg['scriptdirs'] ]
        for path in set(searchdirs):
            if os.path.exists(path):
                files += listTree(path,listdirs=False,sorted=True,includefiles=['.*\.py$'],symlinks=symlinks)
    return files


def grepSource(pattern,options='',relative=True,quiet=False):
    """Finds pattern in the pyFormex source .py files.

    Uses the `grep` program to find all occurrences of some specified
    pattern text in the pyFormex source .py files (including the examples).
    Extra options can be passed to the grep command. See `man grep` for
    more info.
    
    Returns the output of the grep command.
    """
    opts = options.split(' ')
    if '-a' in opts:
        opts.remove('-a')
        options = ' '.join(opts)
        extended = True
    else:
        extended = False
    files = sourceFiles(relative=relative,extended=extended,symlinks=False)
    cmd = "grep %s '%s' %s" % (options,pattern,' '.join(files))
    sta,out = runCommand(cmd,quiet=quiet) 
    return out


###################### locale ###################

def setSaneLocale(localestring=''):
    """Set a sane local configuration for LC_NUMERIC.

    `localestring` is the locale string to be set, e.g. 'en_US.UTF-8'

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

##########################################################################
## Text conversion  tools ##
############################
 
def strNorm(s):
    """Normalize a string.

    Text normalization removes all '&' characters and converts it to lower case.
    """
    return str(s).replace('&','').lower()
   
###################### ReST conversion ###################

#try:
# be quiet, because the import is done early
if checkModule('docutils',quiet=True):
    from docutils.core import publish_string
    def rst2html(text,writer='html'):
        return publish_string(text,writer_name=writer)
#except ImportError:
else:
    def rst2html(text,writer='html'):
        return text
    

def forceReST(text,underline=False):
    """Convert a text string to have it recognized as reStructuredText.

    Returns the text with two lines prepended: a line with '..'
    and a blank line. The text display functions will then recognize the
    string as being reStructuredText. Since the '..' starts a comment in
    reStructuredText, it will not be displayed.

    Furthermore, if `underline` is set True, the first line of the text
    will be underlined to make it appear as a header.
    """
    if underline:
        text = underlineHeader(text)
    return "..\n\n" + text


def underlineHeader(s,char='-'):
    """Underline the first line of a text.

    Adds a new line of text below the first line of s. The new line
    has the same length as the first, but all characters are equal to
    the specified char.
    """
    i = s.find('\n')
    return s[:i] + '\n' + char*i + s[i:]


###################### file conversion ###################

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


def gzip(filename,gzipped=None,remove=True,level=5):
    """Compress a file in gzip format.

    Parameters:

    - `filename`: input file name
    - `gzipped`: output file name. If not specified, it will be set to
      the input file name + '.gz'. An existing output file will be
      overwritten.
    - `remove`: if True (default), the input file is removed after
      succesful compression
    - `level`: an integer from 1..9: gzip compression level. Higher values
      result in smaller files, but require longer compression times. The
      default of 5 gives already a fairly good compression ratio.

    Returns the name of the compressed file.
    """
    import gzip
    if gzipped is None:
        gzipped = filename+'.gz'
    fil = open(filename,'rb')
    gz = gzip.open(gzipped,'wb',compresslevel=level)
    gz.write(fil.read())
    fil.close()
    gz.close()
    if remove:
        removeFile(filename)
    return gzipped


def gunzip(filename,unzipped=None,remove=True):
    """Uncompress a file in gzip format.

    Parameters:

    - `filename`: compressed input file name (usually ending in '.gz')
    - `unzipped`: output file name. If not specified and `filename` ends with
      '.gz', it will be set to the `filename` with the '.gz' removed.
      If an empty string is specified or it is not specified and the filename
      does not end in '.gz', the name of a temporary file is generated. Since
      you will normally want to read something from the decompressed file, this
      temporary file is not deleted after closing. It is up to the user to
      delete it (using the returned file name) when he is ready with it. 
    - `remove`: if True (default), the input file is removed after
      succesful decompression. You probably want to set this to False when
      decompressing to a temporary file.

    Returns the name of the decompressed file.
    """
    import gzip
    gz = gzip.open(filename,'rb')
    if unzipped is None and filename.endswith('.gz'):
        unzipped = filename[:-3]
    if unzipped:
        fil = open(unzipped,'wb')
    else:
        fil = tempFile(prefix='gunzip-',delete=False)
        unzipped = fil.name
    fil.write(gz.read())
    gz.close()
    fil.close()
    if remove:
        removeFile(filename)
    return unzipped


##########################################################################
## File names and formats ##
############################

def all_image_extensions():
    """Return a list with all known image extensions."""
    imgfmt = []
    

file_description = {
    'all': 'All files (*)',
    'ccx': 'CalCuliX files (*.dat *.inp)',
    'dxf': 'AutoCAD .dxf files (*.dxf)',
    'dxfall': 'AutoCAD .dxf or converted(*.dxf *.dxftext)',
    'dxftext': 'Converted AutoCAD files (*.dxftext)',
    'flavia' : 'flavia results (*.flavia.msh *.flavia.res)',
    'gts': 'GTS files (*.gts)',
    'icon': 'Icons (*.xpm)',
    'img': 'Images (*.png *.jpg *.eps *.gif *.bmp)',
    'inp': 'Abaqus or CalCuliX input files (*.inp)',
    'neu': 'Gambit Neutral files (*.neu)',
    'off': 'OFF files (*.off)',
    'pgf': 'pyFormex geometry files (*.pgf)',
    'png': 'PNG images (*.png)',
    'postproc': 'Postproc scripts (*_post.py *.post)',
    'pyformex': 'pyFormex scripts (*.py *.pye)',
    'pyf': 'pyFormex projects (*.pyf)',
    'smesh': 'Tetgen surface mesh files (*.smesh)',
    'stl': 'STL files (*.stl)',
    'surface': 'Surface model (*.off *.gts *.stl *.off.gz *.gts.gz *.stl.gz *.neu *.smesh)',
    'tetgen': 'Tetgen file (*.poly *.smesh *.ele *.face *.edge *.node *.neigh)',
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


def fileType(ftype):
    """Normalize a filetype string.

    The string is converted to lower case and a leading dot is removed.
    This makes it fit for use with a filename extension.

    Example:

    >>> fileType('pdf')
    'pdf'
    >>> fileType('.pdf')
    'pdf'
    >>> fileType('PDF')
    'pdf'
    >>> fileType('.PDF')
    'pdf'
    
    """
    ftype = ftype.lower()
    if len(ftype) > 0 and ftype[0] == '.':
        ftype = ftype[1:]
    return ftype


def fileTypeFromExt(fname):
    """Derive the file type from the file name.

    The derived file type is the file extension part in lower case and
    without the leading dot.

    Example:

    >>> fileTypeFromExt('pyformex.pdf')
    'pdf'
    >>> fileTypeFromExt('.pyformexrc')
    ''
    >>> fileTypeFromExt('pyformex')
    ''
    >>> fileTypeFromExt('pyformex.pgf')
    'pgf'
    >>> fileTypeFromExt('pyformex.pgf.gz')
    'pgf.gz'
    >>> fileTypeFromExt('pyformex.gz')
    'gz'
    """
    name,ext = os.path.splitext(fname)
    ext = fileType(ext)
    if ext == 'gz':
        ext1 = fileTypeFromExt(name)
        if ext1:
            ext = '.'.join([ext1,ext])
    return ext


def findIcon(name):
    """Return the file name for an icon with given name.

    If no icon file is found, returns the question mark icon.
    """
    fname = os.path.join(pf.cfg['icondir'],name) + pf.cfg['gui/icontype']
    if os.path.exists(fname):
        return fname
    return os.path.join(pf.cfg['icondir'],'question') + pf.cfg['gui/icontype']
                                                               

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
    pf.message("Timed evaluation: %s seconds" % (stop-start))
    return res


def countLines(fn):
    """Return the number of lines in a text file."""
    sta,out = runCommand("wc %s" % fn)
    if sta == 0:
        return int(out.split()[0])
    else:
        return 0


##########################################################################
## Running external commands ##
###############################

## def system(cmd):
##     """Execute an external command.

##     This version will write the subprocess stdout and stderr to pyFormex
##     """
##     import subprocess
##     pf.message("Running command: %s" % cmd)
##     P = subprocess.Popen(cmd,shell=True)
##     sta = P.wait() # wait for the process to finish
##     pf.message("Command finished with status: %s" % sta)
##     return sta

def timedWait(proc, timeout, waitToKill = 1.):
    """It is an implementation of the wait() but with a timeout check.
    
    - `timeout`: if a project is not completed before the timeout (float, seconds) it will be terminated.
    - `waitToKill` is the delay between proc.terminate() and proc.kill().
    """
    import timer
    t = timer.Timer(0)    
    while proc.poll() is None and t.seconds(rounded=False) < timeout:#check if proc is completed or timed out
        pass
    if proc.poll() is not None:
        sta = proc.poll() # returncode
        out = proc.communicate()[0] # get the stdout
    elif t.seconds(rounded=False) > timeout:
        proc.terminate()
        try:
            out = proc.stdout.read()
        except:#the stdout was not yet generated
            out = ''
        try:
            import time
            time.sleep(waitToKill)            
            #proc.kill()
            killProcesses([proc.pid+1],signal=15)#VMTK use pid+1, is it general?
        except:
            pass
        sta = -20#time out code (to be decided)
    else:
        raise ValueError,"This really should not happen!"
    return sta, out


def system2(cmd, timeout=None):
    """Execute an external command with timeout.
    
    If timeout is given:
    - `timeout`: if a project is not completed before the timeout (float, seconds) it will be terminated.
    """
    import subprocess
    P = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if timeout == None:
        sta = P.wait() # wait for the process to finish
        out = P.communicate()[0] # get the stdout
    else:
        sta, out = timedWait(P, timeout)  
    return sta,out


# DEPRECATED, KEPT FOR EMERGENCIES
# currently activated by --commands option
def system1(cmd):
    """Execute an external command."""
    import commands
    return commands.getstatusoutput(cmd)


def runCommand(cmd,RaiseError=True,quiet=False,timeout=None):
    """Run a command and raise error if exited with error.

    cmd is a string with the command to be run. The command is run
    in the background, waiting for the result. If no error occurs,
    the exit status and stdout are returned.
    Else an error is raised by default.

    If timeout is given:
    - `timeout`: if a project is not completed before the timeout (float, seconds) it will be terminated.
    
    Example:
    cmd = 'sleep 2'
    sta,out=runCommand3(cmd, RaiseError=False,quiet=False, timeout=5.)
    print (sta,out)
    
    """
    if not quiet:
        pf.message("Running command: %s" % cmd)
    pf.debug("Command: %s" % cmd,pf.DEBUG.INFO)
    #  Use cfg.get here, so that runCommand can be used before condig is done
    if pf.cfg.get('commands',False):
        sta,out = system1(cmd)
    else:
        sta,out = system2(cmd, timeout)
    if sta != 0:
        if not quiet:
            pf.message(out)
            if sta != -20:
                pf.message("Command exited with an error (exitcode %s)" % sta)
            else:
                pf.message("Command exited because timed out")
        if RaiseError:
            
            if sta != -20:
                raise RuntimeError, "Error while executing command:\n  %s" % cmd
            else:
                raise RuntimeError, "Command exited because timed out"
    return sta,out.rstrip('\n')


def spawn(cmd):
    """Spawn a child process."""
    cmd = cmd.split()
    pid = os.spawnvp(os.P_NOWAIT,cmd[0],cmd)
    pf.debug("Spawned child process %s for command '%s'" % (pid,cmd),pf.DEBUG.INFO)
    return pid


def killProcesses(pids,signal=15):
    """Send the specified signal to the processes in list

    - `pids`: a list of process ids.
    - `signal`: the signal to send to the processes. The default (15) will
      try to terminate the process. See 'man kill' for more values.
    """
    for pid in pids:
        try:
            os.kill(pid,signal)
        except:
            pf.debug("Error in killing of process '%s'" % pid,pf.DEBUG.INFO)
            

def changeExt(fn,ext):
    """Change the extension of a file name.

    The extension is the minimal trailing part of the filename starting
    with a '.'. If the filename has no '.', the extension will be appended.
    If the given extension does not start with a dot, one is prepended.

    Example:

    >>> changeExt('image.png','.jpg')
    'image.jpg'
    >>> changeExt('image','.jpg')
    'image.jpg'
    >>> changeExt('image','jpg')
    'image.jpg'
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
    

def userName():
    """Find the name of the user."""
    try:
        return os.environ['LOGNAME']
    except:
        return 'NOBODY'


def is_script(appname):
    """Checks whether an application name is rather a script name"""
    return appname.endswith('.py') or appname.endswith('.pye')

    
def is_app(appname):
    return not is_script(appname)

is_pyFormex = is_script
## def is_pyFormex(filename):
##     """Checks whether a file is a pyFormex script.

##     A file is considered to be a pyFormex script if its name ends in '.py'
##     and the first line of the file contains the substring 'pyformex'.
##     Typically, a pyFormex script starts with a line::

##        # *** pyformex ***
##     """
##     filename = str(filename) # force it into a string
##     if filename.endswith(".pye"):
##         return True

##     ok = filename.endswith(".py")
##     if ok:
##         try:
##             f = open(filename,'r')
##             ok = f.readline().find('pyformex') >= 0
##             f.close()
##         except IOError:
##             ok = False
##     return ok

def getDocString(scriptfile):
    """Return the docstring from a script file.

    This actually returns the first multiline string (delimited by
    triple double quote characters) from the file.
    It does relies on the script file being structured properly and
    indeed including a doctring at the beginning of the file.
    """
    fil = open(scriptfile,'r')
    s = fil.read()
    i = s.find('"""')
    if i >= 0:
        j = s.find('"""',i+1)
        if j >= i+3:
            return s[i+3:j]
    return ''
    

tempFile = tempfile.NamedTemporaryFile
tempDir = tempfile.mkdtemp


def numsplit(s):
    """Split a string in numerical and non-numerical parts.

    Returns a series of substrings of s. The odd items do not contain
    any digits. Joined together, the substrings restore the original.
    The even items only contain digits.
    The number of items is always odd: if the string ends or starts with a
    digit, the first or last item is an empty string.

    Example:

    >>> print(numsplit("aa11.22bb"))
    ['aa', '11', '.', '22', 'bb']
    >>> print(numsplit("11.22bb"))
    ['', '11', '.', '22', 'bb']
    >>> print(numsplit("aa11.22"))
    ['aa', '11', '.', '22', '']
    """
    return digits.split(s)

    
def hsorted(l):
    """Sort a list of strings in human order.

    When human sort a list of strings, they tend to interprete the
    numerical fields like numbers and sort these parts numerically,
    instead of the lexicographic sorting by the computer.

    Returns the list of strings sorted in human order.

    Example:
    >>> hsorted(['a1b','a11b','a1.1b','a2b','a1'])
    ['a1', 'a1.1b', 'a1b', 'a2b', 'a11b']
    """
    def human(s):
        s = digits.split(s)+['0']
        return zip(s[0::2], map(int, s[1::2]))
    return sorted(l,key=human)


def splitDigits(s,pos=-1):
    """Split a string at a sequence of digits.

    The input string is split in three parts, where the second part is
    a contiguous series of digits. The second argument specifies at which
    numerical substring the splitting is done. By default (pos=-1) this is
    the last one.
    
    Returns a tuple of three strings, any of which can be empty. The
    second string, if non-empty is a series of digits. The first and last
    items are the parts of the string before and after that series.
    Any of the three return values can be an empty string.
    If the string does not contain any digits, or if the specified splitting
    position exceeds the number of numerical substrings, the second and
    third items are empty strings.

    Example:

    >>> splitDigits('abc123')
    ('abc', '123', '')
    >>> splitDigits('123')
    ('', '123', '')
    >>> splitDigits('abc')
    ('abc', '', '')
    >>> splitDigits('abc123def456fghi')
    ('abc123def', '456', 'fghi')
    >>> splitDigits('abc123def456fghi',0)
    ('abc', '123', 'def456fghi')
    >>> splitDigits('123-456')
    ('123-', '456', '')
    >>> splitDigits('123-456',2)
    ('123-456', '', '')
    >>> splitDigits('')
    ('', '', '')
    """
    g = numsplit(s)
    n = len(g)
    i = 2*pos
    if i >= -n and i+1 < n:
        if i >= 0:
            i += 1
        #print(g,i,n)
        return ''.join(g[:i]),g[i],''.join(g[i+1:])
    else:
        return s,'',''


# BV: We could turn this into a factory
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

    Example:

    >>> N = NameSequence('abc.98')
    >>> [ N.next() for i in range(3) ]
    ['abc.98', 'abc.99', 'abc.100']
    >>> N = NameSequence('abc-8x.png')
    >>> [ N.next() for i in range(3) ]
    ['abc-8x.png', 'abc-9x.png', 'abc-10x.png']
    >>> NameSequence('abc','.png').next()
    'abc-000.png'
    >>> N = NameSequence('/home/user/abc23','5.png')
    >>> [ N.next() for i in range(2) ]
    ['/home/user/abc235.png', '/home/user/abc245.png']
    """
    
    def __init__(self,name,ext=''):
        """Create a new NameSequence from name,ext."""
        prefix,number,suffix = splitDigits(name)
        if len(number) > 0:
            self.nr = int(number)
            format = "%%0%dd" % len(number)
        else:
            self.nr = 0
            format = "-%03d"
        self.name = prefix+format+suffix+ext

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

    def files(self,sort=hsorted):
        """Return a (sorted) list of files matching the name pattern.

        A function may be specified to sort/filter the list of file names.
        The function should take a list of filenames as input. The output
        of the function is returned. The default sort function will sort
        the filenames in a human order.
        """
        import glob
        files = glob.glob(self.glob())
        if callable(sort):
            files = sort(files)
        return files
    

def prefixDict(d,prefix=''):
    """Prefix all the keys of a dict with the given prefix.

    - `d`: a dict where all the keys are strings.
    - `prefix`: a string

    The return value is a dict with all the items of d, but where the
    keys have been prefixed with the given string.
    """
    return dict([ (prefix+k,v) for k,v in d.items() ])


def subDict(d,prefix='',strip=True):
    """Return a dict with the items whose key starts with prefix.

    - `d`: a dict where all the keys are strings.
    - `prefix`: a string
    - `strip`: if True (default), the prefix is stripped from the keys.
    
    The return value is a dict with all the items from d whose key starts
    with prefix. The keys in the returned dict will have the prefix
    stripped off, unless strip=False is specified.
    """
    if strip:
        return dict([ (k.replace(prefix,'',1),v) for k,v in d.items() if k.startswith(prefix)])
    else:
        return dict([ (k,v) for k,v in d.items() if k.startswith(prefix)])


def selectDict(d,keys):
    """Return a dict with the items whose key is in keys.

    - `d`: a dict where all the keys are strings.
    - `keys`: a set of key values, can be a list or another dict.
    
    The return value is a dict with all the items from d whose key
    is in keys.
    See :func:`removeDict` for the complementary operation.
    
    Example:

    >>> d = dict([(c,c*c) for c in range(6)])
    >>> selectDict(d,[4,0,1])
    {0: 0, 1: 1, 4: 16}
    """
    return dict([ (k,d[k]) for k in set(d)&set(keys) ])


def removeDict(d,keys):
    """Remove a set of keys from a dict.

    - `d`: a dict
    - `keys`: a set of key values

    The return value is a dict with all the items from `d` whose key
    is not in `keys`.
    This is the complementary operation of selectDict.

    Example:

    >>> d = dict([(c,c*c) for c in range(6)])
    >>> removeDict(d,[4,0])
    {1: 1, 2: 4, 3: 9, 5: 25}
    """
    return dict([ (k,d[k]) for k in set(d)-set(keys) ])


def refreshDict(d,src):
    """Refresh a dict with values from another dict.

    The values in the dict d are update with those in src.
    Unlike the dict.update method, this will only update existing keys
    but not add new keys.
    """
    d.update(selectDict(src,d))


def sortedKeys(d):
    """Returns the sorted keys of a dict.

    It is required that the keys of the dict be sortable, e.g. all strings
    or integers.
    """
    k = d.keys()
    k.sort()
    return k

    
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

  
def listFontFiles():
    """List all fonts known to the system.

    Returns a list of path names to all the font files found on the system.
    """
    cmd = 'fc-list : file'
    sta,out = runCommand(cmd)
    if sta:
        warning("fc-list could not find your font files.\nMaybe you do not have fontconfig installed?")
    else:
        return [ f.strip(' :') for f in out.split('\n') ]


###########################################################################
    

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


def inverseDict(d):
    return dict([(v,k) for k,v in d.items()])

_warn_category = { 'U': UserWarning, 'D':DeprecationWarning }

def saveWarningFilter(message,module='',category=UserWarning):
    cat = inverseDict(_warn_category).get(category,'U')
    oldfilters = pf.prefcfg['warnings/filters']
    newfilters = oldfilters + [(str(message),'',cat)]
    pf.prefcfg.update({'filters':newfilters},name='warnings')
    pf.debug("Future warning filters: %s" % pf.prefcfg['warnings/filters'],pf.DEBUG.WARNING)


def filterWarning(message,module='',cat='U',action='ignore'):
    import warnings
    pf.debug("Filter Warning '%s' from module '%s' cat '%s'" % (message,module,cat),pf.DEBUG.WARNING)
    category = _warn_category.get(cat,Warning)
    warnings.filterwarnings(action,message,category,module)


def warn(message,level=UserWarning,stacklevel=3):
    import warnings
    warnings.warn(message,level,stacklevel)


# BEWARE: Do not use yet: DeprecationWarnings are not shown by default
def deprec(message,stacklevel=3):
    warn(message,level=DeprecationWarning,stacklevel=stacklevel)


def deprecation(message):
    def decorator(func):
        def wrapper(*_args,**_kargs):
            warn(message)
            # For some reason these messages are not auto-appended to
            # the filters for the currently running program
            filterWarning(str(message))
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
