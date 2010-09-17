#!/usr/bin/env python
# $Id$
"""Extract info from a Python file and shipout in TeX format.

This script automatically extracts class & function docstrings and argument
list from a module and ships out the information in LaTeX format.

(C) 2009 Benedict Verhegghe (benedict.verhegghe@ugent.be)
I wrote this software in my free time, for my joy, not as a commissioned task.
Any copyright claims made by my employer should therefore be considered void.

It includes parts from the examples in the Python library reference manual
in the section on using the parser module. Refer to the manual for a thorough
discussion of the operation of this code.

Usage:  py2tex.py PYTHONFILE [> outputfile.tex]
"""

import os,sys

import parser
import symbol
import token
import types
from types import ListType, TupleType

# set path to the pyformex modules
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parent)

from pyformex.odict import ODict

def debug(s):
    if options.debug:
        print '.. DEBUG:'+str(s)
        

def get_docs(fileName,dummy=None):
    '''Retrieve information from the parse tree of a source file.

    fileName
        Name of the file to read Python source code from.
    '''
    source = open(fileName).read()
    basename = os.path.basename(os.path.splitext(fileName)[0])
    ast = parser.suite(source)
    return ModuleInfo(ast.totuple(), basename)


class SuiteInfoBase:
    _docstring = ''
    _name = ''
    _arglist = ''

    def __init__(self, tree = None):
        self._class_info = ODict()
        self._function_info = ODict()
        self.arglist = None
        if tree:
            self._extract_info(tree)

    def _extract_info(self, tree):
        # extract docstring
        if len(tree) == 2:
            found, vars = match(DOCSTRING_STMT_PATTERN[1], tree[1])
        else:
            found, vars = match(DOCSTRING_STMT_PATTERN, tree[3])
        if found:
            self._docstring = eval(vars['docstring'])
        # discover inner definitions
        for node in tree[1:]:
            found, vars = match(COMPOUND_STMT_PATTERN, node)
            if found:
                cstmt = vars['compound']
                if cstmt[0] == symbol.funcdef:
                    i = 1
                    class_method = False
                    coords_method = False
                    deprecated = False
                    while i < len(cstmt) and cstmt[i][1] != 'def':
                        if cstmt[i][1][:3] == CLASS_METHOD_PATTERN:
                            class_method = True
                        elif cstmt[i][1][:3] == COORDS_METHOD_PATTERN:
                            coords_method = True
                        elif cstmt[i][1][:3] == DEPRECATED_PATTERN:
                            deprecated = True
                            # LEAVE DEPRECATED FUNCTIONS OUT OF MANUAL
                            break
                        else:
                            #print cstmt[i][1][:3]
                            print ".. POPPING %s" % str(cstmt[i])
                            print ".. MATCHING %s" % str(cstmt[i][1][:3])
                            #raise
                        i += 1
                    # LEAVE DEPRECATED FUNCTIONS OUT OF MANUAL
                    if deprecated:
                        continue
                    if cstmt[i][1] == 'def':
                        name = cstmt[i+1][1]
                        self._function_info[name] = FunctionInfo(cstmt)
                        found, vars = match(DOCSTRING_STMT_PATTERN, tree[1])
                        args = cstmt[i+2][2]
                        self._function_info[name]._arglist = ParameterInfo(args)
                elif cstmt[0] == symbol.classdef:
                    name = cstmt[2][1]
                    self._class_info[name] = ClassInfo(cstmt)

    def get_docstring(self):
        return self._docstring

    def get_name(self):
        return self._name

    def get_class_names(self):
        return self._class_info.keys()

    def get_class_info(self, name):
        return self._class_info[name]

    def __getitem__(self, name):
        try:
            return self._class_info[name]
        except KeyError:
            return self._function_info[name]


class SuiteFuncInfo:
    #  Mixin class providing access to function names and info.

    def get_function_names(self):
        return self._function_info.keys()

    def get_function_info(self, name):
        return self._function_info[name]


class FunctionInfo(SuiteInfoBase, SuiteFuncInfo):
    def __init__(self, tree = None):
        self.class_method = False
        self.coords_method = False
        i = 1
        while i < len(tree) and tree[i][1] != 'def':
            if tree[i][1][:3] == CLASS_METHOD_PATTERN:
                self.class_method = True
            elif tree[i][1][:3] == COORDS_METHOD_PATTERN:
                self.coords_method = True
            else:
                print ".. POPPING %s" % str(tree[i])
            i += 1
            #print i,tree[i+1][1]
        self._name = tree[i+1][1]
        SuiteInfoBase.__init__(self, tree and tree[-1] or None)


class ClassInfo(SuiteInfoBase):
    def __init__(self, tree = None):
        self._name = tree[2][1]
        SuiteInfoBase.__init__(self, tree and tree[-1] or None)

    def get_method_names(self):
        return self._function_info.keys()

    def get_method_info(self, name):
        return self._function_info[name]


class ModuleInfo(SuiteInfoBase, SuiteFuncInfo):
    def __init__(self, tree = None, name = "<string>"):
        self._name = name
        SuiteInfoBase.__init__(self, tree)
        if tree:
            found, vars = match(DOCSTRING_STMT_PATTERN, tree[1])
            if found:
                self._docstring = vars["docstring"]


def match(pattern, data, vars=None):
    """Match `data' to `pattern', with variable extraction.

    pattern
        Pattern to match against, possibly containing variables.

    data
        Data to be checked and against which variables are extracted.

    vars
        Dictionary of variables which have already been found.  If not
        provided, an empty dictionary is created.

    The `pattern' value may contain variables of the form ['varname'] which
    are allowed to match anything.  The value that is matched is returned as
    part of a dictionary which maps 'varname' to the matched value.  'varname'
    is not required to be a string object, but using strings makes patterns
    and the code which uses them more readable.

    This function returns two values: a boolean indicating whether a match
    was found and a dictionary mapping variable names to their associated
    values.
    """
    if vars is None:
        vars = ODict()
    if type(pattern) is ListType:       # 'variables' are ['varname']
        vars[pattern[0]] = data
        return 1, vars
    if type(pattern) is not TupleType:
        return (pattern == data), vars
    if len(data) != len(pattern):
        #print "%s != %s" %  (len(data), len(pattern))
        return 0, vars
    for pattern, data in map(None, pattern, data):
        same, vars = match(pattern, data, vars)
        if not same:
            #print "BREAK AT PATTERN "+str(pattern)
            #print "BREAK WITH DATA  "+str(data)
            break
    return same, vars


#  This pattern identifies compound statements, allowing them to be readily
#  differentiated from simple statements.
#
COMPOUND_STMT_PATTERN = (
    symbol.stmt,
    (symbol.compound_stmt, ['compound'])
    )


#  This pattern will match a 'stmt' node which *might* represent a docstring;
#  docstrings require that the statement which provides the docstring be the
#  first statement in the class or function, which this pattern does not check.
#
DOCSTRING_STMT_PATTERN = (
    symbol.stmt,
    (symbol.simple_stmt,
     (symbol.small_stmt,
      (symbol.expr_stmt,
       (symbol.testlist,
        (symbol.test,
         (symbol.or_test,                # BV added by testing
          (symbol.and_test,
           (symbol.not_test,
            (symbol.comparison,
             (symbol.expr,
              (symbol.xor_expr,
               (symbol.and_expr,
                (symbol.shift_expr,
                 (symbol.arith_expr,
                  (symbol.term,
                   (symbol.factor,
                    (symbol.power,
                     (symbol.atom,
                      (token.STRING, ['docstring'])
                      ))))))))))))))))),
     (token.NEWLINE, '')
     ))


PARAMETER_VALUE_PATTERN = (
    symbol.test,
    (symbol.or_test,
     (symbol.and_test,
      (symbol.not_test,
       (symbol.comparison,
        (symbol.expr,
         (symbol.xor_expr,
          (symbol.and_expr,
           (symbol.shift_expr,
            (symbol.arith_expr,
             (symbol.term,
              (symbol.factor,
               (symbol.power,['parvalue'])
                 ))))))))))))

# The previous does not allow parameters of type DD.bb.cc
# The following does, but rebuildAtom can not yet restore it

PARAMETER_VALUE_PATTERN1 = (
    symbol.test,
    (symbol.or_test,
     (symbol.and_test,
      (symbol.not_test,
       (symbol.comparison,
        (symbol.expr,
         (symbol.xor_expr,
          (symbol.and_expr,
           (symbol.shift_expr,
            (symbol.arith_expr,
             (symbol.term,
              (symbol.factor,['parvalue'])
                 )))))))))))

PARAMETER_VALUE_PATTERN2 = (
    symbol.test,
    (symbol.or_test,
     (symbol.and_test,
      (symbol.not_test,
       (symbol.comparison,
        (symbol.expr,
         (symbol.xor_expr,
          (symbol.and_expr,
           (symbol.shift_expr,
            (symbol.arith_expr,
             (symbol.term,
              (symbol.factor,
               (token.MINUS,'-'),
              (symbol.factor,
               (symbol.power,['parvalue']),
                 )))))))))))))


CLASS_METHOD_PATTERN = (259, (50, '@'), (287, (1, 'classmethod')), )
COORDS_METHOD_PATTERN = (259, (50, '@'), (287, (1, 'coordsmethod')), )
DEPRECATED_PATTERN = (259, (50, '@'), (287, (1, 'deprecated')), )

PARAMETERS_PATTERN = (
    symbol.parameters,
    (token.LPAR, '('),
    (symbol.varargslist, ['arglist']),
    (token.RPAR, ')'),
    )


def rebuildAtom(node):
    if type(node) is TupleType and node[0] == symbol.atom:
        s = ''
        for i in node[1:]:
            debug("S (0) is '%s'" % s) 
            if len(i) == 2 and i[0] < 256:
                # This is a token: just add it
                s += str(i[1])
                debug("S (1) is '%s'" % s)
            elif i[0] == symbol.listmaker:
#                s += '['
#                debug("LIST"+str(i[1:]))
                debug("S (3) is '%s'" % s) 
                for k in i[1:]:
                    s += rebuildAtom(k)
                    debug("S (4) '%s'" % s) 
#                    s += ','
#                    debug("S (5) '%s'" % s) 
#                s += ']'
                debug("S (6) is '%s'" % s) 
            else:
                s += "!!%s!!" % str(i)
                debug("S (7) is '%s'" % s) 
        debug("S (8) is '%s'" % s) 
        return s

    elif node[0] == 12:
        return ','

    elif node[0] == 303:
        return findParameterValue(node)

    if options.error:
        print node
        raise RuntimeError,"ALAS, NOTHING!"
    return ''
       

def findParameterValue(data):
    debug("VALUE TO MATCH " + str(data))
    s = ''

    for i,pattern in enumerate([
        PARAMETER_VALUE_PATTERN,
#        PARAMETER_VALUE_PATTERN1,
        PARAMETER_VALUE_PATTERN2,
        ]):
        debug("TRYING PATTERN %s: %s" %(i,pattern))
        found,vars = match(pattern,data)
        if found:
            if i == 2:
                s += '-'
            break

    if not found:
        if options.error:
            raise RuntimeError,"ALAS, CAN NOT CONTINUE!"
        else:
            print ".. HELP "+str(data)

    return s+rebuildAtom(vars['parvalue'])
    

def ParameterInfo(node):
    #found, vars = match(PARAMETERS_PATTERN, node)
    #print node
    args = []
    name = None
    value = None
    for i in node[1:]:
        if i[0] == symbol.fpdef:
            name = i[1][1]
            value = None
        elif i[0] == 12:
            args.append((name,value))
            name = None
        elif i[0] == 22:
            value = '???'
        elif i[0] == 303:
            value = findParameterValue(i)
        else:
            pass#print i[0]
    if name is not None:
        args.append((name,value))
    #print "ARGS=%s" % args
    return args
        

############# Output formatting ##########################
    

def ship_module(name,docstring):
    shortdoc,longdoc = split_doc(docstring)
    print """.. $%s$  -*- rst -*-
.. pyformex reference manual --- %s
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-%s:

:mod:`%s` --- %s
%s

.. automodule:: %s
   :synopsis: %s

""" % ('Id',name,name,name,shortdoc,'='*(12+len(name)+len(shortdoc)),name,shortdoc)
    

def ship_end():
    print """
   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End
"""


def ship_class_old(name,docstring):
    shortdoc,longdoc = split_doc(docstring)
    print "\n   .. autoclass:: %s\n" % name

def ship_class(name,members=[]):
    print """
   .. autoclass:: %s
      :members: %s""" % (name,','.join(members))

def ship_classinit(name,args,docstring):
    pass
##    print("\nThe %s class has the constructor: \n" % name)

##       :method:
## \\begin{classdesc}{%s}{%s}
## %s

## \\end{classdesc}
## """ % (name,name,ship_args(args),docstring)


def ship_classmethods(name):
    print "\n      %s objects have the following methods:\n" % name 


def ship_method(name,args,docstring,class_method=False,coords_method=False):
    print "      .. automethod:: %s(%s)" % (name,ship_args(args))
    ## print docstring
    ## if coords_method:
    ##     print "\\coordsmethod"
    ## if class_method:
    ##     print "\\classmethod"
    ## print "\\end{funcdesc}\n"
    

def ship_function_section(name):
    s = "Functions defined in the module %s" % name
    print "\n**%s**\n" % s
##     print """
## %s
## %s
## """  % (s,'-'*(len(s)))


def ship_function(name,args,docstring):
    print "   .. autofunction:: %s(%s)" % (name,ship_args(args))


def split_doc(docstring):
    s = docstring.split('\n')
    shortdoc = s[0]
    if len(s) > 2:
        longdoc = '\n'.join(s[2:])
    else:
        longdoc = ''
    return shortdoc.strip('"'),longdoc.strip('"')


def argformat(a):
    if a[1] is None:
        return str(a[0])
    else:
        return '%s=%s' % a

def ship_args(args):
    if len(args) > 0 and args[0][0] == 'self':
        args = args[1:]
    args = map(argformat,args)
    #args = [ a for a in args if a[0] is not None ]
    return sanitize(','.join(args))


def sanitize(s):
    """Sanitize a string for LaTeX."""
    for c in '#&%':
        s = s.replace('\\'+c,c)
        s = s.replace(c,'\\'+c)
    ## for c in '{}':
    ##     s = s.replace('\\'+c,c)
    ##     s = s.replace(c,'\\'+c)
    return s


def do_function(info):
    ship_function(info._name,info._arglist,sanitize(info._docstring))


def do_method(info):
    ship_method(info._name,info._arglist,sanitize(info._docstring),info.class_method,info.coords_method)


def do_class_old(info):
    if info._name.startswith('_'):
        return
    ship_class(info._name,sanitize(info._docstring))
    names = info.get_method_names()
    i = None
    if '__new__' in names:
        i = info['__new__']
    elif '__init__' in names:
        i = info['__init__']
    if i:
        ship_classinit(info._name,i._arglist,sanitize(i._docstring))

    ship_classmethods(info._name)
    for k in names:
        if k == '__init__' or k == '__new__':
            continue
        if k.startswith('_'):
            continue
        do_method(info[k])


def do_class(info):
    if info._name.startswith('_'):
        return
    names = [ n for n in info.get_method_names() if not n.startswith('_') ]
    ship_class(info._name,names)


def do_module(info):
    ship_module(info._name,sanitize(info._docstring))
    for k in info.get_class_names():
        do_class(info[k])

    if len(info.get_class_names()) > 0 and len(info.get_function_names()) > 0:
        ship_function_section(info._name)
    
    for k in info.get_function_names():
        if k.startswith('_'):
            continue
        do_function(info[k])
        
    ship_end()


def main(argv):
    global options
    from optparse import OptionParser,make_option
    parser = OptionParser(
        usage = """usage: %prog [Options] PYTHONFILE
Creates a reference manual in sphinx format for the functions and classes
defined in PYTHONFILE.""",
        version = "%s (C) 2009 Benedict Verhegghe" % __file__,
        option_list=[
        make_option('-d',"--debug", help="print debug info",
                    action="store_true", dest="debug", default=False),
        make_option('-c',"--continue", help="continue on errors",
                    action="store_false", dest="error", default=True),
        ])
    options, args = parser.parse_args(argv)
    
    for a in args:
        info = get_docs(a)
        do_module(info)


if __name__ == "__main__":

    import sys
    
    main(sys.argv[1:])

# End
