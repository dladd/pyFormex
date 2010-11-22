# $Id$

"""Error and Warning Messages

"""

import pyformex as pf


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

warn_viewport_linking = "Linking viewports is an experimental feature and is not fully functional yet."

warn_avoid_sleep = "The sleep function is not yet fully functional and its use should currently be avoided. Use pause or the drawwait setting instead!"

warn_group_tab_items = """The specification of groupboxes or tab forms via a tuple is deprecated!
Please use a dictionary format with itemtype='group' or itemtype='tag',
or use the functions widgets.tabInputItem or widgets.groupInputItem
"""

warn_old_input_dialog = """
InputDialog
-----------
The default InputDialog has changed! See the related help item for more info.
For some time, you will be able to use OldInputDialog to get the old behavior.
We advise you however to convert your program to using the new InputDialog.
"""

warn_old_table_dialog = "The use of OldTableDialog is deprecated. Please use a combination of the Dialog, Tabs and Table widgets."

warn_polyline_directions = "PolyLine.directions() now always returns the same number of directions as there are points. The last direction of an open PolyLine appears twice."

warn_polyline_avgdirections = "PolyLine.avgDirections() now always returns the same number of directions as there are points. For an open PolyLine, the first and last direction are those of the end segment."

warn_quadbezierspline = "The use of the QuadBezierSpline class is deprecated and will be removed in future. Use the BezierSpline class with parameter `degree = 2` instead."

warn_trisurface_getfaces = "TriSurface.getFaces now returns the faces' node numbers. Use TriSurface.getFaceEdges() to get the faces' edge numbers."

# End
