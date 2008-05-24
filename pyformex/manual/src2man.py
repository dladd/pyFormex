#!/bin/env python
#
# Transform a source snippet to manual
#
import re

re_function = re.compile("^def\s+(?P<name>[^\s\(]+)\((?P<args>[^\s\)]+)?\).*")



def dodoc(f):
    """Searches the stream for a doc string"""
    found = False
    for line in f:
        s = line.strip()
        if not found and s.startswith('"""'):
            s = s[3:]
            found = True
        iend = s.find('"""')
        if iend < 0:
            print s
        else:
            print s[:iend]
            return f


def dofunc(f,name,args):
    if args is None:
        args = ''
    print "\n\\begin{funcdesc}{%s}{%s}" % (name,args)
    f = dodoc(f)
    print "\\end{funcdesc}"
    return f
    

def convert(fname):
    f = file(fname,'r')
    for line in f:
        m = re_function.match(line)
        if m is not None:
            f = dofunc(f,m.group('name'),m.group('args'))
        

if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        convert(a)
        
