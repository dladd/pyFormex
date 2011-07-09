# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
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
"""A Python wrapper for unit conversion of physical quantities.

This module uses the standard UNIX program 'units' (available from
http://www.gnu.org/software/units/units.html) to do the actual conversions.
Obviously, it will only work on systems that have this program available.

If you really insist on running another OS lacking the units command,
have a look at http://home.tiscali.be/be052320/Unum.html and make an
implementation based on unum. If you GPL it and send it to me, I might
include it in this distribution.
"""

#from pyformex import utils
#utils.hasExternal('units')

import commands,string
    
def convertUnits(From,To):
    """Converts between conformable units.

    This function converts the units 'From' to units 'To'. The units should
    be conformable. The 'From' argument can (and usually does) include a value.
    The return value is the converted value without units. Thus:
    convertUnits('3.45 kg','g') will return '3450'.
    This function is merely a wrapper around the GNU 'units' command, which
    should be installed for this function to work.
    """
    status,output = commands.getstatusoutput('units \"%s\" \"%s\"' % (From,To))
    if status:
        raise RuntimeError, 'Could not convert units from \"%s\" to \"%s\"' % (From,To) 
    return string.split(output)[1]


class UnitsSystem:
    """A class for handling and converting units of physical quantities.

    The units class provides two built-in consistent units systems:
    International() and Engineering().
    International() returns the standard International Standard units.
    Engineering() returns a consistent engineering system,which is very
    practical for use in mechanical engineering. It uses 'mm' for length
    and 'MPa' for pressure and stress. To keep it consistent however,
    the density is rather unpractical: 't/mm^3'. If you want to use t/m^3,
    you can make a custom units system. Beware with non-consistent unit
    systems though!
    The better practice is to allow any unit to be specified at input
    (and eventually requested for output), and to convert everyting
    internally to a consistent system.
    Apart from the units for usual physical quantities, Units stores two
    special purpose values in its units dictionary:
    'model' : defines the length unit used in the geometrical model
    'problem' : defines the unit system to be used in the problem.
    Defaults are: model='m', problem='international'.
    """
    
    def __init__(self,system='international'):
        self.units = self.Predefined(system)
        self.units['model'] = 'm'
        self.units['problem'] = 'international'


    def Add(self,un):
        """Add the units from dictionary un to the units system"""
        for key,val in un.items():
            self.units[key] = val

    def Predefined(self,system):
        """Returns the predefined units for the specified system"""
        if system == 'international':
            return self.International()
        elif system == 'engineering':
            return self.Engineering()
        elif system == 'user-defined':
            return {}
        else:
            raise RuntimeError,"Undefined Units system '%s'" % system
        

    def International(self):
        """Returns the international units system."""
        return {'length': 'm', 'mass': 'kg', 'force': 'N', 'pressure': 'Pa',
                'density': 'kg/m^3', 'time': 's', 'acceleration': 'm/s^2',
                'temperature': 'tempC', 'degrees': 'K'}


    def Engineering(self):
        """Returns a consistent engineering units system."""
        return {'length': 'mm', 'mass': 't', 'force': 'N', 'pressure': 'MPa',
                'density': 't/mm^3', 'time': 's', 'acceleration': 'mm/s^2',
                'temperature': 'tempC', 'degrees': 'K'}


    def Read(self,filename):
        """Read units from file with specified name.
        
        The units file is an ascii file where each line contains a couple of
        words separated by a colon and a blank. The first word is the type of
        quantity, the second is the unit to be used for this quantity.
        Lines starting with '#' are ignored.
        A 'problem: system' line sets all units to the corresponding value of
        the specified units system.
        """
        fil = file(filename,'r')
        self.units = {}
        for line in fil:
            if line[0] == '#':
                continue
            s = string.split(line)
            if len(s) == 2:
                key,val = s
                key = string.lower(string.rstrip(key,':'))
                self.units[key] = val
                if key == 'problem':
                    self.Add(self.Predefined(string.lower(val)))
            else:
                print("Ignoring line : %s" % line)
        fil.close()

    def Get(self,ent):
        """Get units list for the specified entitities.

        If ent is a single entity, returns the corresponding unit if an entry
        ent exists in the current system or else returns ent unchanged.
        If ent is a list of entities, returns a list of corresponding units.
        Example: with the default units system::
        
          Un = UnitsSystem()
          Un.Get(['length','mass','float'])

        returns: ``['m', 'kg', 'float']``
        """
        if isinstance(ent,list):
            return [ self.Get(e) for e in ent ]
        else:
            if self.units.has_key(ent):
                return self.units[ent]
            else:
                return ent
            

if __name__ == '__main__':

    test = (('21cm','in'),
            ('31e6mg','kg'),
            ('1 lightyear','km'),
            )
    for f,t in test:
        print("%s = %s %s" % (f,convertUnits(f,t),t))

### End
