# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Lindenmayer Systems

"""
from __future__ import print_function

import turtle

class Lima(object):
    """A class for operations on Lindenmayer Systems."""

    def __init__(self,axiom="",rules={}):
        self.axiom = axiom
        self.product = axiom
        self.rule = rules
        self.gen = 0

    def status (self):
        """Print the status of the Lima"""
        print("Lima status:")
        print("  Axiom: %s" % self.axiom)
        print("  Rules: %r" % self.rule)
        print("  Generation: %d" % self.gen)
        print("  Product: %s" % self.product)

    def addRule (self,atom,product):
        """Add a new rule (or overwrite an existing)"""
        self.rule[atom] = product

    def translate (self,rule,keep=False):
        """Translate the product by the specified rule set.

        If keep=True is specified, atoms that do not have a translation
        in the rule set, will be kept unchanged.
        The default (keep=False) is to remove those atoms.
        """
        product = ""
        default = ""
        for c in self.product:
            if keep:
                default=c
            product += rule.get(c,default)
        return product
        
    def grow (self, ngen=1):
        for gen in range(ngen):
            self.product = self.translate(self.rule,keep=True)
            self.gen += 1
        return self.product
    
def lima(axiom,rules,level,turtlecmds,glob=None):
    """Create a list of connected points using a Lindenmayer system.

    axiom is the initial string,
    rules are translation rules for the characters in the string,
    level is the number of generations to produce,
    turtlecmds are the translation rules of the final string to turtle cmds,
    glob is an optional list of globals to pass to the turtle script player.

    This is a convenience function for quickly creating a drawing of a
    single generation member. If you intend to draw multiple generations
    of the same Lima, it is better to use the grow() and translate() methods
    directly.
    """
    A = Lima(axiom,rules)
    A.grow(level)
    scr = "reset();"+A.translate(turtlecmds,keep=False)
    list = turtle.play(scr,glob)
    return list

if __name__ == "__main__":
    def test():
        TurtleRules = { 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' }
        print(lima("F",{"F":"F*F//F*F"},1,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' }))
        print(lima("F",{"F":"F*F//F*F"},2,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })                   )

    test()
    test()
