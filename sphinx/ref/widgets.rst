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

      .. automethod:: getFilename()

   .. autoclass:: ProjectSelection


      ProjectSelection objects have the following methods:

      .. automethod:: getResult()

   .. autoclass:: SaveImageDialog


      SaveImageDialog objects have the following methods:

      .. automethod:: getResult()

   .. autoclass:: ImageViewerDialog


      ImageViewerDialog objects have the following methods:

      .. automethod:: getFilename()

   .. autoclass:: AppearenceDialog


      AppearenceDialog objects have the following methods:

      .. automethod:: setFont()
      .. automethod:: getResult()

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
      .. automethod:: setValue(val)

   .. autoclass:: InputString


      InputString objects have the following methods:

      .. automethod:: show()
      .. automethod:: value()
      .. automethod:: setValue(val)

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

   .. autoclass:: InputDialog


      InputDialog objects have the following methods:

      .. automethod:: timeout()
      .. automethod:: timedOut()
      .. automethod:: show(timeout=None,timeoutfunc=None,modal=False)
      .. automethod:: acceptData(result=ACCEPTED)
      .. automethod:: updateData(d)
      .. automethod:: getResult(timeout=None)

   .. autoclass:: TableModel


      TableModel objects have the following methods:

      .. automethod:: rowCount(parent=None)
      .. automethod:: columnCount(parent=None)
      .. automethod:: data(index,role)
      .. automethod:: headerData(col,orientation,role)
      .. automethod:: insertRows(row=None,count=None)
      .. automethod:: removeRows(row=None,count=None)

   .. autoclass:: Table


      Table objects have the following methods:


   .. autoclass:: TableDialog


      TableDialog objects have the following methods:


   .. autoclass:: ButtonBox


      ButtonBox objects have the following methods:

      .. automethod:: setText(text,index=0)
      .. automethod:: setIcon(icon,index=0)

   .. autoclass:: ComboBox


      ComboBox objects have the following methods:

      .. automethod:: setIndex(i)

   .. autoclass:: BaseMenu


      BaseMenu objects have the following methods:

      .. automethod:: item(text)
      .. automethod:: itemAction(item)
      .. automethod:: insert_sep(before=None)
      .. automethod:: insert_menu(menu,before=None)
      .. automethod:: insert_action(action,before=None)
      .. automethod:: create_insert_action(str,val,before=None)
      .. automethod:: insertItems(items,before=None)

   .. autoclass:: Menu


      Menu objects have the following methods:

      .. automethod:: process()
      .. automethod:: remove()

   .. autoclass:: MenuBar


      MenuBar objects have the following methods:


   .. autoclass:: DAction


      DAction objects have the following methods:

      .. automethod:: activated()

   .. autoclass:: ActionList


      ActionList objects have the following methods:

      .. automethod:: add(name,icon=None)
      .. automethod:: names()

**Functions defined in the module widgets**

   .. autofunction:: selectFont()
   .. autofunction:: getColor(col=None,caption=None)
   .. autofunction:: inputAny(name,value,itemtype=str)
   .. autofunction:: inputAnyOld(item,parent=None)
   .. autofunction:: updateDialogItems(data,newdata)
   .. autofunction:: dialogButtons(dialog,actions,default=None)
   .. autofunction:: messageBox(message,level='info',choices=['OK'],default=None,timeout=None)
   .. autofunction:: textBox(text,type=None,choices=['OK'])
   .. autofunction:: normalize(s)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

