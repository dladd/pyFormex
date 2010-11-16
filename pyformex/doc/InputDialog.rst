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

All input items are now dictionaries. The plugin.widgets module
however provides some functions which ease the construction of these
dictionaries from a sequence of arguments.

It is convenient to import these functions in the following way, providing
short aliases for use in your code. ::
  
  from plugins.widgets import simpleInputItem as I, \
                              groupInputItem as G, \
                              tabInputItem as T
                                
Of course you only need to import those function that you really need

How to convert your old style dialogs to the new style ?
--------------------------------------------------------

The following guidelines hold whether you are working directly with the
``gui.widgets.InputDialog`` or you are using the ``askItems()`` function.

Input items which are just (name,value) tuples can be converted by simply
placing an I in front, turning the tuple into a function call, which will
create a proper InputItem dictionary out of the data. If you had a list 
instead of a tuple, you will have to replace the brackets with parentheses.
All other item fields can be entered in the function call as keyword
arguments.

Suppose you have the following in old style ::
  
  res = askItems([
      ('width',w),
      ['height',h],
      ('angle',a,{'text':'Rotation angle'}),
      ('color',None,'radio',{'choices':['red',green','blue']}),
      ])

This can be converted to the following new style specification ::
  
  res = askItems([
      I('width',w),
      I('height',h),
      I('angle',a,text='Rotation angle'),
      I('color',itemtype='radio',choices=['red',green','blue']),
      ])

Use the 'G' to create a grouped set of input items, and 'T' to create 
tabbed pages of input items. Both should receive a list of input items as
their value (second argument).


Transitional facilities
-----------------------

compatInputItem
...............
To ease the transition to the new dialog format, the following transitional 
facilities are provided. They will help you to run your existing programs
with a minimum of changes. Remember though that these facilities are scheduled
to fade out before pyFormex reaches version 1.0!

The ``compatInputItem`` function will accept most old style item input
and create a proper item dictionary. Thus, the above old style dialog
could have been changed to ::
  
  from plugins.widgets import compatInputItem as C
  res = askItems([
      C('width',w),
      C('height',h),
      C('angle',a,{'text':'Rotation angle'}),
      C('color',None,'radio',{'choices':['red',green','blue']}),
      ])

and it would work again. So, if you are in a hurray, all you need to do is
prepend the 'C' and change brackets to parentheses. 

Automatic conversion with askItems
..................................

The function askItems will now by default try to 
automatically do the above conversion, even if you did not add the 'C'. Thus 
you could have stayed with ::
  
  res = askItems([
      ('width',w),
      ('height',h),
      ('angle',a,{'text':'Rotation angle'}),
      ('color',None,'radio',{'choices':['red',green','blue']}),
      ])

where you only changed the bracktes in the second item.  This
automatic conversion is not available if you directly use the
InputDialog widget. Also, the conversion is not guaranteed to work for
every old input format. If needed, you can swithch it off by
specifying an extra argument: ``legacy=False`` will not try conversion
and assume everything is specified in new style; ``legacy=True`` will
assume everything is specified in old style.


Final remark
------------

Remember that the transitional facilities will be removed in future. Do not
wait too long to make a proper conversion to the new style.

.. End
