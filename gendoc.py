#!/usr/bin/env python
# Id$

import os,sys,imp,pyclbr
from pyformex import odict

p = os.path.join(os.path.dirname(__file__),'pyformex')
print "Adding '%s' to search path" %p
sys.path[0:0] = [p]


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


class Exporter():
    def ship_header(self,*args):
        pass
    def ship_footer(self,*args):
        pass
    def ship_module(self,*args):
        pass
    def ship_class_header(self,*args):
        pass
    def ship_class(self,*args):
        pass
    def ship_function_header(self,*args):
        pass
    def ship_function(self,*args):
        pass


class TextExporter(Exporter):
    def ship_header(self,h):
        print "==== %s ====" % h
    def ship_footer(self):
        print "=" * 40
    def ship_class_header(self,ch):
        print "CLASSES"
    def ship_function_header(self,fh):
        print "FUNCTIONS"
    def ship_module(self,m):
        print "MODULE: %s" % m.__name__
        print m.__doc__
    def ship_class(self,k,v):
        print k
        print v.__dict__
    def ship_function(self,k,v):
        print k
        print v.__dict__
    


def dodoc(d,m,f):
    f.ship_header("CONTENTS OF MODULE %s" % m.__name__)
    f.ship_module(m)
    d = sortDict(d)
    classes,functions = splitDict(d)
    f.ship_class_header(m)
    for k,v in classes.items():
        f.ship_class(k,v)
    f.ship_function_header(m)
    for k,v in functions.items():
        f.ship_function(k,v)
    f.ship_footer()
    

def gendoc(a):
    fn = a + '.py'
    if not os.path.exists(fn):
        print "File %s does not exist" % fn
        return -1

    m = imp.load_source(a,fn)
    d = pyclbr.readmodule_ex(a)

    fn = a + '_doc.tex'
    f = file(fn,'w')
    res = 1
    try:
        #sys.stdout = f
        exporter = TextExporter()
        dodoc(d,m,exporter)
        res = 0
    finally:
        sys.stdout = sys.__stdout__
        f.close()
    return res



if __name__ == "__main__":

    import sys
    for a in sys.argv[1:]:
        print "Processing %s" % a
        gendoc(a)
