#!/usr/bin/env python
# $Id$
#
"Lindenmayer Systems"

# Convenience functions: trigonometric functions with argument in degrees
# Should we keep this in here???

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
        """Add a new rule (or overwrite an exisiting)"""
        self.rule[atom] = product

    def translate (self,rule,keep=False):
        """Translate the product by the specified rule set.

        If keep=True is specified, atoms that do not have a translation
        in the rule set, will be kept unchanged.
        The default (keep=False) is to reove those atoms.
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
    
def lima(axiom,rules,level,turtle):
    """Create a list of connected points using a Lindenmayer system."""
    import Turtle
    A = Lima(axiom,rules)
    A.grow(level)
    #A.status()
    scr = "reset();"+A.translate(turtle,keep=False)
    #print(scr)
    list = Turtle.play(scr)
    #print len(list)," lines"
    return list

if __name__ == "__main__":
    def test():
        TurtleRules = { 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' }
        print lima("F",{"F":"F*F//F*F"},1,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })
        print lima("F",{"F":"F*F//F*F"},2,{ 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' })                   

    test()
    test()
