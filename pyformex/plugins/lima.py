#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"Lindenmayer Systems"

import turtle

class Lima:
    """A class for operations on Lindenmayer Systems."""

    def __init__(self,axiom="",rules={}):
        self.axiom = axiom
        self.product = axiom
        self.rule = rules
        self.gen = 0

    def status (self):
        """Print the status of the Lima"""
        print "Lima status:"
        print "  Axiom: %s" % self.axiom
        print "  Rules: %r" % self.rule
        print "  Generation: %d" % self.gen
        print "  Product: %s" % self.product

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
        default=""
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
        print lima("F",{"F":"F*F//F*F"},1,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })
        print lima("F",{"F":"F*F//F*F"},2,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })                   

    test()
    test()
