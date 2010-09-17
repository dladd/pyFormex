.. $Id$  -*- rst -*-
.. pyformex reference manual --- widgets
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-widgets:

:mod:`widgets` --- A collection of custom widgets used in the pyFormex GUI
==========================================================================

.. automodule:: widgets
   :synopsis: A collection of custom widgets used in the pyFormex GUI



   .. autoclass:: Options
      :members: 

   .. autoclass:: FileSelection
      :members: show,getFilename

   .. autoclass:: ProjectSelection
      :members: getResult

   .. autoclass:: SaveImageDialog
      :members: getResult

   .. autoclass:: ImageViewerDialog
      :members: getFilename

   .. autoclass:: DockedSelection
      :members: setSelected,getResult

   .. autoclass:: ModelessSelection
      :members: setSelected,getResult

   .. autoclass:: Selection
      :members: setSelected,getResult

   .. autoclass:: InputItem
      :members: setTooltip,name,text,value,setValue

   .. autoclass:: InputInfo
      :members: value

   .. autoclass:: InputString
      :members: show,value

   .. autoclass:: InputText
      :members: show,value,setValue

   .. autoclass:: InputBool
      :members: text,value,setValue

   .. autoclass:: InputCombo
      :members: value,setValue

   .. autoclass:: InputRadio
      :members: value,setValue

   .. autoclass:: InputPush
      :members: setText,setIcon,value,setValue

   .. autoclass:: InputInteger
      :members: show,value,setValue

   .. autoclass:: InputFloat
      :members: show,value,setValue

   .. autoclass:: InputFloatTable
      :members: show,value,setValue

   .. autoclass:: InputSlider
      :members: set_value

   .. autoclass:: InputFSlider
      :members: set_value

   .. autoclass:: InputColor
      :members: setColor,setValue

   .. autoclass:: InputFont
      :members: setFont

   .. autoclass:: InputWidget
      :members: text,value,setValue

   .. autoclass:: InputGroup
      :members: value,setValue

   .. autoclass:: NewInputDialog
      :members: add_items,add_tab,add_group,add_input,timeout,timedOut,show,acceptData,updateData,getResult

   .. autoclass:: OldInputDialog
      :members: add_input_items,timeout,timedOut,show,acceptData,updateData,getResult

   .. autoclass:: InputDialog
      :members: 

   .. autoclass:: TableModel
      :members: makeEditable,rowCount,columnCount,data,headerData,insertRows,removeRows,flags,setData

   .. autoclass:: ArrayModel
      :members: setData

   .. autoclass:: Table
      :members: 

   .. autoclass:: Tabs
      :members: 

   .. autoclass:: Dialog
      :members: add

   .. autoclass:: TableDialog
      :members: 

   .. autoclass:: OldTableDialog
      :members: 

   .. autoclass:: MessageBox
      :members: show,getResult,updateText

   .. autoclass:: TextBox
      :members: getResult,updateText

   .. autoclass:: InputBox
      :members: 

   .. autoclass:: ButtonBox
      :members: setText,setIcon

   .. autoclass:: ComboBox
      :members: setIndex

   .. autoclass:: CoordsBox
      :members: setValues

**Functions defined in the module widgets**

   .. autofunction:: setInputTimeout(timeout)
   .. autofunction:: addTimeOut(widget,timeout=???,timeoutfunc=???)
   .. autofunction:: selectFont()
   .. autofunction:: getColor(col=???,caption=???)
   .. autofunction:: defaultItemType(item)
   .. autofunction:: simpleInputItem(name,value=???,itemtype=???)
   .. autofunction:: groupInputItem(name,items=???)
   .. autofunction:: tabInputItem(name,items=???)
   .. autofunction:: compatInputItem(name,value,itemtype=???,kargs=???)
   .. autofunction:: convertInputItemList(items)
   .. autofunction:: inputAny(name,value,itemtype=???)
   .. autofunction:: inputAnyOld(item,parent=???)
   .. autofunction:: updateDialogItems(data,newdata)
   .. autofunction:: updateText(widget,text,format=???)
   .. autofunction:: dialogButtons(dialog,actions,default=???)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

