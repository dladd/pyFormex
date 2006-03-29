#!/usr/bin/env python
# $Id: $

"""Create a Formex from a file with coordinates"""
 
from formex import *

def splitCoords(s, sep=','):
    """Split a string in 3 floats (coordinates) on occurrence of sep.

    The string is split on the occurrence of the substring sep.
    The substrings are then converted to floats, and combined in a list [c1,c2,c3]
    If there is no sep, no value is returned.
    """
    n = s.find(sep)
    if n >= 0:
        c1 = float(s[:n])
        s2 = s[n+len(sep):]
        n2 = s2.find(sep)
        if n2 >= 0:
            c2 = float(s2[:n2])
            c3 = float(s2[n2+len(sep):])
            return ([c1,c2,c3])
    else:
        print ('Seperator %s was not found'%sep)
        #return ( s, '' )

def fileFormex(fil, sep=',', closed='No'):
    """Reads coordinates from a file and creates a Formex.
   
    The coordinates in the file are seperated by sep.
    """
    
    
    inp=open(fil,'r')
    allcoords=inp.readlines()
    F=Formex([[splitCoords(allcoords[1],sep),splitCoords(allcoords[2], sep)]])
    for l in range(len(allcoords)-1)[0:]:
        point1=splitCoords(allcoords[l],sep)
        point2=splitCoords(allcoords[l+1],sep)
        Fh=Formex([[point1,point2]])
        F.append(Fh)
    if closed.upper()=='YES':
        G=Formex([[splitCoords(allcoords[len(allcoords)-1],sep),splitCoords(allcoords[0], sep)]])
        F.append(G)
    return F


if __name__ == '__main__':

##    test = open ('testfile', 'r')
##    fil = test.readline()
##    a=splitCoords(fil, ',')
##    print a	     
    G=fileFormex('testfile')
    print G
    H=fileFormex('testfile', closed='yes')
    print H
