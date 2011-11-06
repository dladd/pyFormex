#
##
##  This file is part of the pyFormex project.
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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
#
/^python:/ { print "python="$2 }
/^ansic:/ { print "ansic="$2 }
/^sh:/ { print "sh="$2 }
/^Total.*(SLOC)/ { sub("[^=]*= ",""); sub(",",""); print "sloc="$0 }
/^Devel.*Months)/ { sub("[^=]*= ",""); sub(",",""); print "manyears="$1 }
/^Schedule.*Months)/ { sub("[^=]*= ",""); sub(",",""); print "years="$1 }
/^Total Estimated Cost/ { sub("[^=]*= ",""); gsub(",",""); print "dollars="$2 }
