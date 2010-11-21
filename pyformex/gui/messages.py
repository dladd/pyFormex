# $Id$

"""Error and Warning Messages

"""

import pyformex as pf
## import warnings

## def warning(msg,ver=''):
##     msg = globals().get(msg,msg)
##     warnings.warn(msg)


def getMessage(msg):
    """Return the real message corresponding with the specified mnemonic.

    If no matching message was defined, the original is returned.
    """
    msg = str(msg) # allows for msg being a Warning
    return globals().get(msg,msg)


warn_askitems_changed = """.. warn_askitems_changed

askItems
--------
The default operation of askItems has changed!
It will now by default try to convert the items to use the new InputDialog.

The old InputDialog will still be available for some time by using the
'legacy = True' argument, but we advice you to switch to the newer InputItem
format as soon as possible.

Using 'legacy = False' will force the use of the new format.

The default 'legacy=None' tries to convert old data when they are found and
when they are convertible.
"""

warn_drawaxes_changed = "The syntax of drawAxes has changed. The use of the 'pos' argument is deprecated. Use an appropriate CoordinateSystem instead."

warn_viewport_switching = """.. warn_viewport_switching

Viewport switching
------------------
The viewport switching functions have changed: interactive changes through the
GUI are now decoupled from changes by the script.
This may result in unwanted effects if your script relied on the old (coupled)
functionality.

If you notice any unexpected behaviour, please tell the developers about it
through the `forums <%s>`_ or `bug system <%s>`_.
""" % (pf.cfg['help/forums'],pf.cfg['help/bugs'])
  
# End
