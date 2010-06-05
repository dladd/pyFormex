.. $Id$  -*- rst -*-
.. pyformex reference manual --- widgets
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-widgets:

:mod:`widgets` --- A collection of custom widgets used in the pyFormex GUI
==========================================================================

.. automodule:: widgets
   :synopsis: A collection of custom widgets used in the pyFormex GUI



   .. autoclass:: Options


      Options objects have the following methods:


   .. autoclass:: FileSelection


      FileSelection objects have the following methods:

      .. automethod:: show(timeout=None,timeoutfunc=None,modal=False)
      .. automethod:: getFilename(timeout=None)

   .. autoclass:: ProjectSelection


      ProjectSelection objects have the following methods:

      .. automethod:: getResult()

   .. autoclass:: SaveImageDialog


      SaveImageDialog objects have the following methods:

      .. automethod:: getResult()

   .. autoclass:: ImageViewerDialog


      ImageViewerDialog objects have the following methods:

      .. automethod:: getFilename()

   .. autoclass:: DockedSelection


      DockedSelection objects have the following methods:

      .. automethod:: setSelected(selected,bool)
      .. automethod:: getResult()

   .. autoclass:: ModelessSelection


      ModelessSelection objects have the following methods:

      .. automethod:: setSelected(selected,bool)
      .. automethod:: getResult()

   .. autoclass:: Selection


      Selection objects have the following methods:

      .. automethod:: setSelected(selected)
      .. automethod:: getResult()

   .. autoclass:: InputItem


      InputItem objects have the following methods:

      .. automethod:: name()
      .. automethod:: text()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputInfo


      InputInfo objects have the following methods:

      .. automethod:: value()

   .. autoclass:: InputString


      InputString objects have the following methods:

      .. automethod:: show()
      .. automethod:: value()

   .. autoclass:: InputText


      InputText objects have the following methods:

      .. automethod:: show()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputBool


      InputBool objects have the following methods:

      .. automethod:: text()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputCombo


      InputCombo objects have the following methods:

      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputRadio


      InputRadio objects have the following methods:

      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputPush


      InputPush objects have the following methods:

      .. automethod:: setText(text,index=0)
      .. automethod:: setIcon(icon,index=0)
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputInteger


      InputInteger objects have the following methods:

      .. automethod:: show()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputFloat


      InputFloat objects have the following methods:

      .. automethod:: show()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputFloatTable


      InputFloatTable objects have the following methods:

      .. automethod:: show()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputSlider


      InputSlider objects have the following methods:

      .. automethod:: set_value(val)

   .. autoclass:: InputFSlider


      InputFSlider objects have the following methods:

      .. automethod:: set_value(val)

   .. autoclass:: InputColor


      InputColor objects have the following methods:

      .. automethod:: setColor()
      .. automethod:: setValue(value)

   .. autoclass:: InputFont


      InputFont objects have the following methods:

      .. automethod:: setFont()

   .. autoclass:: InputWidget


      InputWidget objects have the following methods:

      .. automethod:: text()
      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: InputGroup


      InputGroup objects have the following methods:

      .. automethod:: value()
      .. automethod:: setValue(val)

   .. autoclass:: NewInputDialog


      NewInputDialog objects have the following methods:

      .. automethod:: add_items(items,form)
      .. automethod:: add_tab(name,items)
      .. automethod:: add_group(name,items)
      .. automethod:: add_input(item)
      .. automethod:: timeout()
      .. automethod:: timedOut()
      .. automethod:: show(timeout=None,timeoutfunc=None,modal=False)
      .. automethod:: acceptData(result=ACCEPTED)
      .. automethod:: updateData(d)
      .. automethod:: getResult(timeout=None)

   .. autoclass:: OldInputDialog


      OldInputDialog objects have the following methods:

      .. automethod:: add_input_items(items,layout)
      .. automethod:: timeout()
      .. automethod:: timedOut()
      .. automethod:: show(timeout=None,timeoutfunc=None,modal=False)
      .. automethod:: acceptData(result=ACCEPTED)
      .. automethod:: updateData(d)
      .. automethod:: getResult(timeout=None)

   .. autoclass:: InputDialog


      InputDialog objects have the following methods:


   .. autoclass:: TableModel


      TableModel objects have the following methods:

      .. automethod:: makeEditable(edit=True)
      .. automethod:: rowCount(parent=None)
      .. automethod:: columnCount(parent=None)
      .. automethod:: data(index,role)
      .. automethod:: headerData(col,orientation,role)
      .. automethod:: insertRows(row=None,count=None)
      .. automethod:: removeRows(row=None,count=None)
      .. automethod:: flags(index)
      .. automethod:: setData(index,value,role=_EDITROLE)

   .. autoclass:: ArrayModel


      ArrayModel objects have the following methods:

      .. automethod:: setData(index,value,role=_EDITROLE)

   .. autoclass:: Table


      Table objects have the following methods:


   .. autoclass:: Tabs


      Tabs objects have the following methods:


   .. autoclass:: Dialog


      Dialog objects have the following methods:

      .. automethod:: add(widgets,pos=1)

   .. autoclass:: TableDialog


      TableDialog objects have the following methods:


   .. autoclass:: OldTableDialog


      OldTableDialog objects have the following methods:


   .. autoclass:: MessageBox


      MessageBox objects have the following methods:

      .. automethod:: show(modal=False)
      .. automethod:: getResult()
      .. automethod:: updateText(text,format='')

   .. autoclass:: TextBox


      TextBox objects have the following methods:

      .. automethod:: getResult()
      .. automethod:: updateText(text,format='')

   .. autoclass:: InputBox


      InputBox objects have the following methods:


   .. autoclass:: ButtonBox


      ButtonBox objects have the following methods:

      .. automethod:: setText(text,index=0)
      .. automethod:: setIcon(icon,index=0)

   .. autoclass:: ComboBox


      ComboBox objects have the following methods:

      .. automethod:: setIndex(i)

   .. autoclass:: CoordsBox


      CoordsBox objects have the following methods:

      .. automethod:: setValues(values)

**Functions defined in the module widgets**

   .. autofunction:: setInputTimeout(timeout)
   .. autofunction:: addTimeOut(widget,timeout=None,timeoutfunc=None)
   .. autofunction:: selectFont()
   .. autofunction:: getColor(col=None,caption=None)
   .. autofunction:: defaultItemType(item)
   .. autofunction:: simpleInputItem(name,value=None,itemtype=None)
   .. autofunction:: groupInputItem(name,items=[])
   .. autofunction:: tabInputItem(name,items=[])
   .. autofunction:: compatInputItem(name,value,itemtype=None,kargs={})
   .. autofunction:: inputAny(name,value,itemtype=str)
   .. autofunction:: inputAnyOld(item,parent=None)
   .. autofunction:: updateDialogItems(data,newdata)
   .. autofunction:: updateText(widget,text,format='')
   .. autofunction:: dialogButtons(dialog,actions,default=None)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

