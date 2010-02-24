.. $Id$
        
=====================
Change in InputDialog
=====================
The syntax of InputDialog items will change in version 0.9.
The new syntax is already available with the classname 'NewInputDialog'.
After the change, the old syntax will still be available for some time as
'OldInputDialog'.
We advice you however to move to the new syntax as soon as possible, so that
you do not get into troubles when the OldInputDialog is removed.

Here are some guidelines to ease the conversion:

- If you are using the askItems() function, you should not make any changes.
- If you use widgets.InputDialog() directly, you can either:

  - replace it with OldInputDialog, and this message will go away, 
    but you're in for a future breakage.
  - convert your InputItems to the new format: you're safe for the future,
    but it might take you some work now.
  - Or you can take the intermediate (lazy) path, and use the
    widgets.compatInputItem() function to transform old item syntax to new
    one. This only works for simple InputItem structures though. Group boxes
    and tabbed pages will need to be changed manually.

Here's an example of how you can easily make the transition for simple
InputDialogs.

Say that you have the following InputDialog::

    dia = widgets.InputDialog(items=[
        ('width',w),
	('height',h),
	])

The widgets.compatInputItem() function can transform a single InputItem 
from the old (tuple) format to the new (dict) format. If you import that
function under a short alias, the changes become minimal::

    from gui.widgets import compatInputItem as C

    dia = widgets.NewInputDialog(items=[
        C('width',w),
	C('height',h),
	])

.. End
