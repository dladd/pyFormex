#!/usr/bin/env pyformex
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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
from simple import *
from numpy import *

class DxfExporter(object):
    """Export geometry in DXF format.

    While we certainly do not want to promote proprietary software,
    some of our users occasionally needed to export some model in
    DXF format.
    This class provides a minimum of functionality.
    """

    def __init__(self,filename,terminator='\n'):
        """Open a file for export in DXF format.

        No check is done that the file has a '.dxf' extension.
        The file will by default be written in UNIX line termination mode.
        An existing file will be overwritten without warning!
        """
        self.filename = filename
        self.fil = file(self.filename,'w')
        self.term = terminator


    def write(self,s):
        """Write a string to the dxf file.

        The string does not include the line terminator.
        """
        self.fil.write(s+self.term)
        

    def out(self,code,data):
        """Output a string data item to the dxf file.

        code is the group code,
        data holds the data
        """
        self.write('%3s' % code)
        self.write('%s' % data)

        
    def close(self):
        """Finalize and close the DXF file"""
        self.out(0,'EOF')
        self.fil.close()
        
        
    def section(self,name):
        """Start a new section"""
        self.out(0,'SECTION')
        self.out(2,name)


    def endSection(self):
        """End the current section"""
        self.out(0,'ENDSEC')
        

    def entities(self):
        """Start the ENTITIES section"""
        self.section('ENTITIES')


    def layer(self,layer):
        """Export the layer"""
        self.out(8,layer)


    def line(self,x,layer=0):
        """Export a line.

        x is a (2,3) shaped array
        """
        self.out(0,'LINE')
        self.out(8,layer)
        for j in range(2):
            for i in range(3):
                self.out(10*(i+1)+j,x[j][i])


def exportDXF(filename,F):
    """Export a Formex to a DXF file

    Currently, only plex-2 Formices can be exported to DXF.
    """
    if F.nplex() != 2:
        raise ValueError,"Can only export plex-2 Formices to DXF"
    dxf = DxfExporter(filename)
    dxf.entities()
    for i in F:
        dxf.line(i)
    dxf.endSection()
    dxf.close()


# An example

if __name__ == 'draw':
    chdir(__file__)
    c = circle(360./20.,360./20.,360.)
    draw(c)
    exportDXF('circle1.dxf',c)

#End
