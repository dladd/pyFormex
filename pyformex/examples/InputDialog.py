#!/usr/bin/env pyformex
# $Id$

"""InputDialog

Example showing the use of the InputDialog

level = 'normal'
topics = []
techniques = ['dialog']

"""

from gui.widgets import simpleInputItem as I, groupInputItem as G, tabInputItem as T


# BV
# not working correctly:
# - itemtype = 'label'
# - bounds on 'float'
# - fslider has no slider
# - tooltip on group/tab

input_text = [
    I('info','A constant info field',text='itemtype info',itemtype='info',tooltip='This is an informational field that can not be changed'),
#    I('label','A constant info field',text='itemtype label',itemtype='label',tooltip='This is an informational field that can not be changed'),
    I('string','A string input field',text='itemtype string',itemtype='string',tooltip='This is a single line string input field'),
    I('text','A multiline text input field',text='itemtype text',itemtype='text',tooltip='This is a multiline text input field'),
    ]
input_select = [
    I('bool',False,text='itemtype bool',tooltip='This is a boolean field that can only get the values True or False, by checking or unchecking the box'),
    I('select','Third',text='itemtype select',choices=['First','Second','Third','Fourth'],tooltip='This is a an input field allowing you to select one of a set of predefined values'),
    I('radio','Third',text='itemtype (h)radio',itemtype='radio',choices=['First','Second','Third','Fourth'],tooltip="Like 'select', this allows selecting one of a set of predefined values"),
    I('vradio','Third',text='itemtype vradio',itemtype='vradio',choices=['First','Second','Third','Fourth'],tooltip="Like 'radio', but items are placed vertically"),
    I('push','Third',text='itemtype (h)push',itemtype='push',choices=['First','Second','Third','Fourth'],tooltip="Yet another method to select one of a set of predefined values"),
    I('vpush','Third',text='itemtype vpush',itemtype='vpush',choices=['First','Second','Third','Fourth'],tooltip="Like 'push', but items are placed vertically"),
    ]
input_numerical = [
    I('integer',37,text='any integer',tooltip='An integer input field'),
    I('bounded',3,text='a bounded integer (0..10)',min=0,max=10,tooltip='A bounded integer input field. This value is e.g. bounded to the interval [0,10]'),
    I('float',37,text='any float',tooltip='An integer input field'),
    I('boundedf',23.7,text='a bounded float',min=23.5,max=23.9,tooltip='A bounded float input field. This value is e.g. bounded to the interval [23.5,23.9]'),
    I('slider',3,text='a integer slider',min=0,max=10,itemtype='slider',tooltip='An integer input field accompanied by a slider to set the value.'),
    I('fslider',23.7,text='a float slider',min=23.5,max=23.9,itemtype='fslider',tooltip='A float input field accompanied by a slider to set the value.'),
    ]

input_special = [
    I('color',colors.pyformex_pink,itemtype='color',text='Color',tooltip='An inputfield allowing to select a color. The current color is pyFormex pink.'),
    ]
##     G('Custom Color palette',[
##         I('maxcol',[1.,0.,0.],text='Maximum color'),
##         I('medcol',[1.,1.,0.],text='Medium color'),
##         I('mincol',[1.,1.,1.],text='Minimum color'),
##         ],enabled=False),
##     I('font','hv18',text='Font',choices=GLUTFONTS.keys()),

input_data = [
    T('Text',input_text),
    T('Selection',input_select),
    T('Numerical',input_numerical),
    T('Special',input_special),
    ]

input_enablers = [
##     ('valrange','Minimum-Medium-Maximum','medval','medcol'),
##     ('predef',True,'palet'),
##     ('predef',False,'Custom Color palette'),
##     ('showgrid',True,'linewidth'),
    ]


def show():
    """Accept the data and show the results"""
    dialog.acceptData()
    res = dialog.results
    print res


def close():
    global dialog
    pf.PF['ColorScale_data'] = dialog.results
    if dialog:
        dialog.close()
        dialog = None


def timeOut():
    """What to do on a InputDialog timeout event.

    As a policy, all pyFormex examples should behave well on a
    dialog timeout.
    Most users can simply ignore this.
    """
    show()
    close()


if __name__ == 'draw':

    # Update the data items from saved values
    try:
        saved_data = named('InputDialog_data')
        widgets.updateDialogItems(input_data,save_data)
    except:
        pass

    # Create the modeless dialog widget
    dialog = widgets.InputDialog(input_data,enablers=input_enablers,caption='ColorScale Dialog',actions = [('Close',close),('Show',show)],default='Show')

    # Examples style requires a timeout action
    dialog.timeout = timeOut

    # Show the dialog and let the user have fun
    dialog.show()

# End

