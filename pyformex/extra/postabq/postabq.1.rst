=======
postabq
=======

------------------------------------------------------
Postprocess results from an Abaqus(C) simulation
------------------------------------------------------

:Author: postabq was written by Benedict Verhegghe <benedict.verhegghe@ugent.be> for the pyFormex project. This manual page was written for the Debian project (and may be used by others).
:Date:   2012-08-08
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

SYNOPSIS
========

postabq [OPTIONS] FILE

DESCRIPTION
===========

postabq scans an Abaqus output file (.fil format) and converts the data
into a Python script that can be interpreted by the pyFormex fe_post plugin.
The Python script is written to stdout.

The postabq command is usually installed under the name 'pyformex-postabq'.

OPTIONS
=======

-v  Be verbose (mostly for debugging)
-e  Force EXPLICIT from the start (default is to autodetect)
-n  Dry run: run through the file but do not produce conversion
-h  Print this help text
-V  Print version and exit

SEE ALSO
========

The pyFormex project <https://savannah.nongnu.org/projects/pyformex/>.
