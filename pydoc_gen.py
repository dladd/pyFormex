#!/usr/bin/env python
# 
"""Generate documentation for Python modules.

This uses the pydoc2 module as e.g. found with OpenGLContext.
"""

Version = "0.1 (C) Benedict Verhegghe"

#from OpenGLContext.pydoc import pydoc2
import pydoc2
from optparse import OptionParser,make_option


def main(argv=None):

    # this allows us to call main from the interpreter
    if argv is None:
        argv = sys.argv

    # Process options
    parser = OptionParser(
        usage = "usage: %prog [<options>] module ...",
        version = Version,
        option_list=[
        make_option('-x',"--excludes", help="A list of modules to be excluded. It should be a single string with the modules sepeated by blanks.",
                    action="store", dest="exclude", default=''),
        make_option("-d","--dir", help="Destination directory for the output.",
                    action="store", dest="dir", default='.'),
        ])
    options, args = parser.parse_args()

    print options.exclude.split()

    pydoc2.PackageDocumentationGenerator(
        baseModules = args,
        exclusions = options.exclude.split(),
        destinationDirectory = options.dir,
        ).process ()
 
#### Go

if __name__ == "__main__":
    import sys
    sys.exit(main())

#### End
