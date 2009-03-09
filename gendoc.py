#!/usr/bin/env python
# Id$

__copyright__ = "(C) 2009 Benedict Verhegghe"
__version__ = "gendoc.py 0.1"


import os,sys,imp,pyclbr,re
from pyformex import odict

#re_arglist = re.compile("^\s?def\s(?P<name>[^\s(])\s?(?P<args>[^)])\)\w?:")
re_arglist = re.compile("^\s?def\s(?P<name>[^\s(]*)\s?\((?P<args>.*)\)\w?:")


def print_do(s):
    """Print a string"""
    print s
def print_skip(s):
    pass


prt = print_skip


def sanitize(s):
    """Sanitize a string for LaTeX."""
    if s is None:
        return ''
    s = s.replace('\#','#')
    s = s.replace('#','\#')
    s = s.replace('\&','&')
    s = s.replace('&','\&')
    return s

def addScriptPath():
    """Add the script path to the search path"""
    p = os.path.join(os.path.dirname(__file__),'pyformex')
    prt("Adding '%s' to search path" % p)
    sys.path[0:0] = [p]


def splitDocString(docstring):
    s = docstring.split('\n')
    shortdoc = s[0]
    if len(s) > 2:
        longdoc = '\n'.join(s[2:])
    else:
        longdoc = ''
    return shortdoc.strip('"'),longdoc.strip('"')


def sortDict(d):
    """Sort a module dictionary based on the lineno.

    Returns an ODict with the items of d in order of their occurrence
    in the source file.
    """
    items = d.items()
    items.sort(lambda x,y: x[1].lineno - y[1].lineno)
    return odict.ODict(items)


def splitDict(d):
    """Split a module dictionary in class and function definitions.

    Returns two ODicts: one with the classes and another with the functions,
    each in order of their occurrence in the supplied dict.
    """
    items = d.items()
    classes = [ x for x in items if isinstance(x[1],pyclbr.Class) ]
    functions = [ x for x in items if isinstance(x[1],pyclbr.Function) ]
    return odict.ODict(classes),odict.ODict(functions)


def scanForArgs(fn,lnrs):
    """Scan for argument lists on line lineno of file f."""
    argd = {}
    if len(lnrs) > 0:
        f = file(fn,'r')
        lnrs = [ i-1 for i in lnrs ]
        lnr = lnrs.pop(0)
        for i,line in enumerate(f):
            if i < lnr:
                continue
            else:
                #print i,line
                m = re_arglist.match(line)
                if m is None:
                    print "Argument list not found on line %s of file %s" % (i+1,fn)
                else:
                    argd[i+1] = m.groupdict()
                if len(lnrs) == 0:
                    break
                lnr = lnrs.pop(0)
        f.close()
    return argd
            
    

############################################################################
class Module(object):
    
    def __init__(self,fullname):
        fn = fullname + '.py'
        if not os.path.exists(fn):
            raise ValueError,"Module source file %s does not exist" % fn

        m = imp.load_source(fullname,fn)
        d = pyclbr.readmodule_ex(fullname)
        # remove external defs
        d = dict([ (k,v) for k,v in d.items() if v.module == fullname ])
        d = sortDict(d)
    
        self.module = m
        self.filename = fn
        self.fullname = fullname
        self.name = self.fullname.split('/')[-1]
        self.shortdoc,self.longdoc = splitDocString(m.__doc__)
        # get classes and functions
        self.classes,self.functions = splitDict(d)
        prt("Classes: %s" % ', '.join(self.classes.keys()))
        prt("Functions: %s" % ', '.join(self.functions.keys()))
        self.get_function_arguments()


    def get_function_arguments(self):
        # get line numbers
        lnrs = [ f.lineno for f in self.functions.values() ]
        args = scanForArgs(self.filename,lnrs)
        #print args
        # check consistency
        for n,f in self.functions.items():
            a = args.get(f.lineno,None)
            if a is None or a['name'] != n:
                print args
                print f.module
                raise ValueError,"Argument list not found for function %s on line %s" % (n,f.lineno)
            f.arglist = a['args']
            

    def output(self,formatter):
        out = ""
        out += formatter.header(self.name)
        out += formatter.module_header(self.name,self.fullname,self.shortdoc,self.longdoc)
        if len(self.classes) > 0:
            out += formatter.classes_header(self.name)
            for k,v in self.classes.items():
                prt("Class %s" % k)
        if len(self.functions) > 0:
            out += formatter.functions_header(self.name)
            for k,v in self.functions.items():
                prt("Function %s" % k)
                doc = sanitize(self.module.__dict__[k].__doc__)
                out += formatter.function_doc(k,v.arglist,doc)
        out += formatter.footer(self.name)
        return out


    
##########################################################################

class Formatter(object):
    @staticmethod
    def header(self,name):
        return ''
    @staticmethod
    def footer(self,name):
        return ''
    @staticmethod
    def module_header(name,fullname,shortdoc,longdoc):
        return ''
    @staticmethod
    def classes_header(name):
        return ''
    @staticmethod
    def functions_header(name):
        return ''
    @staticmethod
    def class_doc():
        return ''
    @staticmethod
    def function_doc(name,args,doc):
        return ''


class TextFormatter(Formatter):
    @staticmethod
    def header(name):
        return "============= %s ============" % name
    @staticmethod
    def footer(name):
        return "=" * 40
    @staticmethod
    def module_header(name,fullname,shortdoc,longdoc):
        return "MODULE: %s" % name
    @staticmethod
    def classes_header(name):
        return "CLASSES:"
    @staticmethod
    def functions_header(name):
        return "FUNCTIONS:"
    @staticmethod
    def class_doc(self,k,v):
        return k
        return v.__dict__
    @staticmethod
    def function_doc(self,k,v):
        return k
        return v.__dict__


class LaTeXFormatter(Formatter):
    @staticmethod
    def header(name):
        return """%% pyformex manual --- %s
%% $Id$
%% (C) B.Verhegghe

""" % name

    @staticmethod
    def footer(name):
        return """
%%%%%% Local Variables: 
%%%%%% mode: %s
%%%%%% TeX-master: "pyformex"
%%%%%% End:
""" % "latex"

    @staticmethod
    def module_header(name,fullname,shortdoc,longdoc):
        return """
\\section{\\module{%s} --- %s}
\\label{sec:%s}

\\declaremodule{""}{%s}
\\modulesynopsis{%s}
\\moduleauthor{'pyFormex project'}{'http://pyformex.org'}

%s
""" % (fullname,shortdoc,name,fullname,fullname,longdoc)

    @staticmethod
    def classes_header(name):
        return ""

    @staticmethod
    def functions_header(name):
        return """
\\subsection{Functions defined in module \module{%s}}
""" % name

    @staticmethod
    def function_doc(name,args,doc):
        return """
\\begin{funcdesc}{%s}{%s}
%s

\\end{funcdesc}
""" % (name,args,doc)


##     def classinit(name,args,docstring):
##         print """
##     \\begin{classdesc}{%s}{%s}
##     %s

##     \\end{classdesc}
##     """ % (name,args(args),docstring)


##     def class(name,docstring):
##         print """
##     \\subsection{%s class}

##     %s
##     """ % (name,docstring)



    

def gendoc(a,formatter):
    prt("\nModule: %s" % a)

    M = Module(a)

    fn = options.output.replace('%f',M.fullname).replace('%n',M.name)
    if not options.force:
        if os.path.exists(fn):
            print "File %s exists already: use --force option to overwrite" % fn
            return
        
    prt("Writing output file %s" % fn)
    f = file(fn,'w')
    res = 1
    try:
        f.write(M.output(formatter))
        res = 0
    finally:
        f.close()
    return res


def main(argv):
    global options,prt
    from optparse import OptionParser,make_option

    default_output = "pyformex/manual/ref-%n.tex"
    parser = OptionParser(
        usage = """usage: %prog [<options>] modulename ...]
  modulename is a Python module or package/module that can be imported from
  the search path.
  Module names can be given with or without a trailing '.py'.""",
        version = __version__,
        option_list=[
        make_option("--text", help="Set output format to ASCII (default)",
                    action="store_const", const="text", dest="format", default='text'),
        make_option("--latex", help="Set output format to LaTeX",
                    action="store_const", const="latex", dest="format"),
        make_option("--output", help="Set the name of the output file. The value may contain a substring '%f' or '%n', which will be replaced by the full model name or module name",
                    action="store", dest="output",default=default_output),
        make_option("--force", help="Force overwriteing of existing output files. By default, modules for which the output file exists will be skipped.",
                    action="store_true", dest="force", default=False),
        make_option("--verbose",'-v', help="Print out information on the processing.",
                    action="store_true", dest="verbose", default=False),
        ])
        
    options, args = parser.parse_args(argv)
    if options.verbose:
        prt = print_do

    addScriptPath()
    
    if options.format == 'latex':
        formatter = LaTeXFormatter
    else:
        formatter = TextFormatter
    
    for a in args:
        gendoc(a,formatter)


if __name__ == "__main__":

    import sys
    main(sys.argv[1:])
