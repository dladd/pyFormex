#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

"""InputDialog

Example showing the use of the InputDialog

level = 'normal'
topics = []
techniques = ['dialog']

"""


# BV
# not working correctly:
# - itemtype = 'label'
# - bounds on 'float'
# - fslider has no slider
# - tooltip on group/tab
# - push does not work correctly

input_text = [
    dict(name='label',value=None,text="A constant info field",itemtype='info',tooltip="itemtype='info' with a value=None can be used to show a label only. Note that this does not work with the _I() function."),
    _I('info','A constant info field',text="itemtype 'info'",itemtype='info',tooltip="This is an informational field that can not be changed"),
    _I('string','A string input field',text="itemtype str",itemtype='string',tooltip="This is a single line string input field"),
    _I('text','A multiline text input field',text="itemtype 'text'",itemtype='text',tooltip="This is a multiline text input field"),
    ]

input_select = [
    _I('bool',False,text="itemtype bool",stretch='ba',tooltip="This is a boolean field that can only get the values True or False, by checking or unchecking the box"),
    _I('list',['First','Third'],text="itemtype 'list'",itemtype='list',choices=['First','Second','Third','Fourth'],tooltip="This is a an input field allowing you to select one or more of a set of predefined values"),
    _I('slist',['First','Third'],text="itemtype 'list' with single=True",itemtype='list',choices=['First','Second','Third','Fourth'],single=True,tooltip="This is like 'list', bu allowing only a single value to be selected"),
    _I('clist',['First','Third'],text="itemtype 'list' with check=True",itemtype='list',choices=['First','Second','Third','Fourth'],check=True,tooltip="This is a an input field allowing you to select one of a set of predefined values"),
    _I('select','Third',text="itemtype 'select'",choices=['First','Second','Third','Fourth'],tooltip="This is a an input field allowing you to select one of a set of predefined values"),
    _I('radio','Third',text="itemtype (h)radio",itemtype='radio',choices=['First','Second','Third','Fourth'],tooltip="Like 'select', this allows selecting one of a set of predefined values"),
    _I('vradio','Third',text="itemtype vradio",itemtype='vradio',choices=['First','Second','Third','Fourth'],tooltip="Like 'radio', but items are placed vertically"),
#    _I('push','Third',text="itemtype (h)push",itemtype='push',choices=['First','Second','Third','Fourth'],tooltip="Yet another method to select one of a set of predefined values"),
#    _I('vpush','Third',text="itemtype vpush",itemtype='vpush',choices=['First','Second','Third','Fourth'],tooltip="Like 'push', but items are placed vertically"),
    ]

input_numerical = [
    _I('integer',37,text="any int",tooltip="An integer input field"),
    _I('bounded',3,text="a bounded integer (0..10)",min=0,max=10,tooltip="A bounded integer input field. This value is e.g. bounded to the interval [0,10]"),
    _I('float',37.,text="any float",tooltip="A float input field"),
    _I('boundedf',23.7,text="a bounded float",min=23.5,max=23.9,tooltip="A bounded float input field. This value is e.g. bounded to the interval [23.5,23.9]"),
    _I('slider',3,text="a integer slider",min=0,max=10,itemtype='slider',tooltip="An integer input field accompanied by a slider to set the value."),
    _I('fslider',23.7,text="a float slider",min=23.5,max=23.9,itemtype='fslider',tooltip="A float input field accompanied by a slider to set the value."),
    _I('ivector',[640,400],text="an integer vector",tooltip="An integer vector input field"),
    ]

input_special = [
    _I('color',colors.pyformex_pink,itemtype='color',text="Color",tooltip="An inputfield allowing to select a color. The current color is pyFormex pink."),
    _I('font','',itemtype='font'),
    _I('point',[0.,0.,0.],itemtype='point'),
    ]

input_tabgroup = [
    _I('enable1',False,text="Enable group 1"),
    _G('group1',input_text,text="Text input group"),
    _I('enable2',False,text="Enable group 2"),
    _G('group2',input_select,text="Select input group"),
    _G('group3',input_special,text="Special input group",checkable=True,checked=True),
    ]

input_data = [
    _T('Text',input_text),
    _T('Selection',input_select),
    _T('Numerical',input_numerical),
    _T('Special',input_special),
    _T('tabgroup',input_tabgroup,text="Tabs and Groups"),
    ]

input_enablers = [
    ('tabgroup/enable1',True,'tabgroup/group1','tabgroup/group2/radio'),
    ('tabgroup/enable2',True,'tabgroup/group2','tabgroup/group3','tabgroup/group1/info'),
    ]


def show():
    """Accept the data and show the results"""
    from utils import formatDict
    dialog.acceptData()
    res = dialog.results
    print formatDict(res)


def close():
    global dialog
    pf.PF['ColorScale_data'] = dialog.results
    if dialog:
        dialog.close()
        dialog = None
    # Release scriptlock
    scriptRelease(__file__)


def timeOut():
    """What to do on a InputDialog timeout event.

    As a policy, all pyFormex examples should behave well on a
    dialog timeout.
    Most users can simply ignore this.
    """
    show()
    wait()
    close()


if __name__ == 'draw':

    # Update the data items from saved values
    try:
        saved_data = named('InputDialog_data')
        widgets.updateDialogItems(input_data,save_data)
    except:
        pass

    # Create the modeless dialog widget
    dialog = widgets.InputDialog(input_data,enablers=input_enablers,autoprefix=True,caption='InputDialog',actions = [('Close',close),('Show',show)],default='Show')

    # Examples style requires a timeout action
    dialog.timeout = timeOut

    # Show the dialog and let the user have fun
    dialog.show()

    # Block other scripts 
    scriptLock(__file__)

# End

