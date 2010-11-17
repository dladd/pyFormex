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
   :members: setInputTimeout,addTimeOut,selectFont,getColor,defaultItemType,simpleInputItem,groupInputItem,tabInputItem,compatInputItem,convertInputItemList,inputAny,inputAnyOld,updateDialogItems,updateText,dialogButtons

   ``Classes defined in module widgets``


   .. autoclass:: Options
      :members: 

   .. autoclass:: FileSelection
      :members: show,getFilename

   .. autoclass:: ProjectSelection
      :members: getResult

   .. autoclass:: SaveImageDialog
      :members: getResult

   .. autoclass:: DockedSelection
      :members: setSelected,getResult

   .. autoclass:: ModelessSelection
      :members: setSelected,getResult

   .. autoclass:: Selection
      :members: setSelected,getResult

   .. autoclass:: InputItem
      :members: name,text,value,setValue

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

   .. autoclass:: InputPoint
      :members: value,setValue

   .. autoclass:: InputButton
      :members: doFunc

   .. autoclass:: InputColor
      :members: setColor,setValue

   .. autoclass:: InputFont
      :members: setFont

   .. autoclass:: InputWidget
      :members: text,value,setValue

   .. autoclass:: InputGroup
      :members: value,setValue

   .. autoclass:: NewInputDialog
      :members: add_items,add_tab,add_group,add_input,timeout,timedOut,show,acceptData,updateData,getResults

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
      :members: getValues,setValues

   .. autoclass:: ImageView
      :members: showImage

   ``Functions defined in module widgets`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

