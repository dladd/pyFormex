# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""A collection of custom widgets used in the pyFormex GUI

The widgets in this module were primarily created in function of the
pyFormex GUI. The user can apply them to change the GUI or to add
interactive widgets to his scripts. Of course he can also use all the
Qt widgets directly.
"""

import os,types
from PyQt4 import QtCore, QtGui
import pyformex as GD
import colors
import odict
import imageViewer
import utils


# timeout value for all widgets providing timeout feature
#  (currently: InputDialog, MessageBox)

input_timeout = -1  # default timeout value : -1 means no timeout

def setInputTimeout(timeout):
    global input_timeout
    input_timeout = timeout
    

# result values for dialogs
ACCEPTED = QtGui.QDialog.Accepted
REJECTED = QtGui.QDialog.Rejected
TIMEOUT = -1        # the return value if a widget timed out

# slots
Accept = QtCore.SLOT("accept()")
Reject = QtCore.SLOT("reject()")

def addTimeOut(widget,timeout=None,timeoutfunc=None):
    """Add a timeout to a widget.

    - `timeoutfunc` is a callable. If None it will be set to the widget's
      `timeout` method if one exists.
    - `timeout` is a float value. If None, it will be set to to the global
      `input_timeout`.

    If timeout is positive, a timer will be installed into the widget which
    will call the `timeoutfunc` after `timeout` seconds have elapsed.
    The `timeoutfunc` can be any callable, but usually will emit a signal
    to make the widget accept or reject the input. The timeoutfunc will not
    be called is if the widget is destructed before the timer has finished.
    """
    if timeout is None:
        timeout = input_timeout
    if timeoutfunc is None and hasattr(widget,'timeout'):
        timeoutfunc = widget.timeout

    try:
        timeout = float(timeout)
        if timeout >= 0.0:
            GD.debug("ADDING TIMEOUT %s,%s" % (timeout,timeoutfunc))
            timer = QtCore.QTimer()
            if type(timeoutfunc) is str:
                timer.connect(timer,QtCore.SIGNAL("timeout()"),widget,QtCore.SLOT(timeoutfunc))
            else:
                timer.connect(timer,QtCore.SIGNAL("timeout()"),timeoutfunc)
            timer.setSingleShot(True)
            timeout = int(1000*timeout)
            timer.start(timeout)
            widget.timer = timer  # make sure this timer stays alive
            GD.debug("TIMER STARTED")
    except:
        raise ValueError,"Could not start the timeout timer"



class Options:
    pass

###################### File Selection Dialog #########################

class FileSelection(QtGui.QFileDialog):
    """A file selection dialog.

    You can specify a default path/filename that will be suggested initially.
    If a pattern is specified, only matching files will be shown.
    A pattern can be something like ``Images (*.png *.jpg)`` or a list
    of such strings.
    Default mode is to accept any filename. You can specify exist=True
    to accept only existing files. Or set exist=True and multi=True to
    accept multiple files.
    If dir==True, a single existing directory is asked.
    """
    
    def __init__(self,path,pattern=None,exist=False,multi=False,dir=False):
        """The constructor shows the widget."""
        QtGui.QFileDialog.__init__(self)
        if os.path.isfile(path):
            self.setDirectory(os.path.dirname(path))
            self.selectFile(path)
        else:
            self.setDirectory(path)
        if type(pattern) == str:
            self.setFilter(pattern)
        else: # should be a list of patterns
            self.setFilters(pattern)
        if dir:
            mode = QtGui.QFileDialog.Directory
            caption = "Select a directory"
        elif exist:
            if multi:
                mode = QtGui.QFileDialog.ExistingFiles
                caption = "Select existing files"
            else:
                mode = QtGui.QFileDialog.ExistingFile
                caption = "Open existing file"
        else:
            mode = QtGui.QFileDialog.AnyFile
            caption = "Save file as"
        self.setFileMode(mode)
        self.setWindowTitle(caption)
        if exist:
            self.setLabelText(QtGui.QFileDialog.Accept,'Open')
        else:
            self.setLabelText(QtGui.QFileDialog.Accept,'Save')
##         if self.sidebar:
##             urls = self.sidebarUrls()
##             for f in self.sidebar:
##                 urls.append(QtCore.QUrl.fromLocalFile(f))
##             self.setSidebarUrls(urls)
##         for p in self.sidebarUrls():
##             GD.message(p.toString())

    timeout = "accept()"

    def show(self,timeout=None,timeoutfunc=None,modal=False):
        self.setModal(modal)
        QtGui.QFileDialog.show(self)
        addTimeOut(self,timeout,timeoutfunc)
        
    def getFilename(self,timeout=None):
        """Ask for a filename by user interaction.

        Return the filename selected by the user.
        If the user hits CANCEL or ESC, None is returned.
        """
        self.show(timeout,modal=True)
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            files = map(str,self.selectedFiles())
            if self.fileMode() == QtGui.QFileDialog.ExistingFiles:
                return files
            else:
                return files[0]
        else:
            return None


class ProjectSelection(FileSelection):
    """A file selection dialog specialized for opening projects."""
    def __init__(self,path=None,pattern=None,exist=False,compression=4,ignore_signature=True):
        """Create the dialog."""
        if path is None:
            path = GD.cfg['workdir']
        if pattern is None:
            pattern = map(utils.fileDescription, ['pyf'])  
        FileSelection.__init__(self,path,pattern,exist)
        grid = self.layout()
        nr,nc = grid.rowCount(),grid.columnCount()

        if not exist:
            self.cpr = InputSlider("Compression level (0-9)",compression,min=0,max=9)
            self.cpw = QtGui.QWidget()
            self.cpw.setLayout(self.cpr)
            self.cpw.setToolTip("Higher compression levels result in smaller files, but higher load and save times.")
            grid.addWidget(self.cpw,nr,0,1,-1)
            nr += 1

        if exist:
            self.sig = QtGui.QCheckBox("Ignore Signature Version")
            if ignore_signature:
                self.sig.setCheckState(QtCore.Qt.Checked)
            self.sig.setToolTip("Check this box to allow opening projects saved with an older version number in the header.")
            grid.addWidget(self.sig,nr,0,1,-1)
            nr += 1

            self.leg = QtGui.QCheckBox("Allow Opening Legacy Format")
            self.leg.setToolTip("Check this box to allow opening projects saved in the headerless legacy format.")
            grid.addWidget(self.leg,nr,0,1,-1)
            nr += 1


    def getResult(self):
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            opt = odict.ODict()
            opt.fn = str(self.selectedFiles()[0])
            opt.cpr = opt.sig = opt.leg = None
            if hasattr(self,'cpr'):
                opt.cpr = self.cpr.value()
            if hasattr(self,'leg'):
                opt.leg = self.leg.isChecked()
            if hasattr(self,'sig'):
                opt.sig = self.sig.isChecked()
            return opt
        else:
            return None


class SaveImageDialog(FileSelection):
    """A dialog for saving to an image file.

    The dialog contains the normal file selection widget plus some
    extra fields to set the Save Image parameters.
    """
    def __init__(self,path=None,pattern=None,exist=False,multi=False):
        """Create the dialog."""
        if path is None:
            path = GD.cfg['workdir']
        if pattern is None:
            pattern = map(utils.fileDescription, ['img','icon','all'])  
        FileSelection.__init__(self,path,pattern,exist)
        grid = self.layout()
        nr,nc = grid.rowCount(),grid.columnCount()
        self.win = QtGui.QCheckBox("Whole Window")
        self.roo = QtGui.QCheckBox("Crop Root")
        self.bor = QtGui.QCheckBox("Add Border")
        self.mul = QtGui.QCheckBox("Multi mode")
        self.hot = QtGui.QCheckBox("Activate '%s' hotkey" % GD.cfg['keys/save'])
        self.aut = QtGui.QCheckBox('Autosave mode')
        self.mul.setChecked(multi)
        self.hot.setChecked(multi)
        self.win.setToolTip("If checked, the whole window is saved;\nelse, only the Canvas is saved.")
        self.roo.setToolTip("If checked, the window will be cropped from the root window.\nThis mode is required if you want to include the window decorations.")
        self.bor.setToolTip("If checked when the whole window is saved,\nthe window decorations will be included as well.")
        self.mul.setToolTip("If checked, multiple images can be saved\nwith autogenerated names.")
        self.hot.setToolTip("If checked, a new image can be saved\nby hitting the 'S' key when focus is in the Canvas.")
        self.aut.setToolTip("If checked, a new image will saved\non each draw() operation")
        grid.addWidget(self.win,nr,0)
        grid.addWidget(self.roo,nr,1)
        grid.addWidget(self.bor,nr,2)
        nr += 1
        grid.addWidget(self.mul,nr,0)
        grid.addWidget(self.hot,nr,1)
        grid.addWidget(self.aut,nr,2)

    def getResult(self):
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            opt = Options()
            opt.fn = str(self.selectedFiles()[0])
            opt.wi = self.win.isChecked()
            opt.rc = self.roo.isChecked()
            opt.bo = self.bor.isChecked()
            opt.mu = self.mul.isChecked()
            opt.hk = self.hot.isChecked()
            opt.au = self.aut.isChecked()
            return opt
        else:
            return None


class ImageViewerDialog(QtGui.QDialog):
    """A dialog to select an image file.

    The dialog helps selecting the correct image by displaying the
    image in an image viewer widget.
    """
    def __init__(self,path=None):
        QtGui.QDialog.__init__(self)
        box = QtGui.QHBoxLayout()
        self.viewer = imageViewer.ImageViewer(parent=self,path=path)
        box.addWidget(self.viewer)
        self.setLayout(box)
        
    def getFilename(self):
        """Ask for a filename by user interaction.

        Return the filename selected by the user.
        If the user hits CANCEL or ESC, None is returned.
        """
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            return str(self.viewer.filename)
        else:
            return None
        self.close()
        

def selectFont():
    """Ask the user to select a font.

    A font selection dialog widget is displayed and the user is requested
    to select a font.
    Returns a font if the user exited the dialog with the :guilabel:`OK`
    button.
    Returns None if the user clicked :guilabel:`CANCEL`.
    """
    font,ok = QtGui.QFontDialog.getFont()
    if ok:
        return font
    else:
        return None

        
class AppearenceDialog(QtGui.QDialog):
    """A dialog for setting the GUI appearance.

    The dialog lets you select one of the Qt styles.
    """
    
    def __init__(self):
        """Create the Appearance dialog."""
        self.font = None
        QtGui.QDialog.__init__(self)
        self.setWindowTitle('Appearance Settings')
        # Style
        styleLabel = QtGui.QLabel('Style')
        self.styleCombo = QtGui.QComboBox()
        styles = map(str,QtGui.QStyleFactory().keys())
        GD.debug("Available styles : %s" % styles)
        style = GD.app.style().objectName()
        GD.debug("Current style : %s" % style)
        self.styleCombo.addItems(styles)
        self.styleCombo.setCurrentIndex([i.lower() for i in styles].index(style))
        # Font
        fontLabel = QtGui.QLabel('Font')
        font = GD.app.font().toString()
        GD.debug("Current font : %s" % font)
        self.fontButton = QtGui.QPushButton(font)
        self.connect(self.fontButton,QtCore.SIGNAL("clicked()"),self.setFont)
        # Accept/Cancel Buttons
        acceptButton = QtGui.QPushButton('OK')
        self.connect(acceptButton,QtCore.SIGNAL("clicked()"),self,Accept)
        cancelButton = QtGui.QPushButton('Cancel')
        self.connect(cancelButton,QtCore.SIGNAL("clicked()"),self,Reject)
        # Putting it all together
        grid = QtGui.QGridLayout()
        grid.setColumnStretch(1,1)
        grid.setColumnMinimumWidth(1,250)
        grid.addWidget(styleLabel,0,0)
        grid.addWidget(self.styleCombo,0,1,1,2)
        grid.addWidget(fontLabel,1,0)
        grid.addWidget(self.fontButton,1,1,1,-1)
        grid.addWidget(acceptButton,2,3)
        grid.addWidget(cancelButton,2,4)
        self.setLayout(grid)


    def setFont(self):
        font = selectFont()
        if font:
            self.fontButton.setText(font.toString())
            self.font = font


    def getResult(self):
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            style = QtGui.QStyleFactory().create(self.styleCombo.currentText())
            return style,self.font 
        else:
            return None,None


class DockedSelection(QtGui.QDockWidget):
    """A widget that is docked in the main window and contains a modeless
    dialog for selecting items.
    """
    def __init__(self,slist=[],title='Selection Dialog',mode=None,sort=False,func=None):
        QtGui.QDockWidget.__init__(self)
        self.setWidget(ModelessSelection(slist,title,mode,sort,func))
    
    def setSelected(self,selected,bool):
        self.widget().setSelected(selected,bool)
    
    def getResult(self):
        res = self.widget().getResult()
        return res


class ModelessSelection(QtGui.QDialog):
    """A modeless dialog for selecting one or more items from a list."""
    
    selection_mode = {
        None: QtGui.QAbstractItemView.NoSelection,
        'single': QtGui.QAbstractItemView.SingleSelection,
        'multi': QtGui.QAbstractItemView.MultiSelection,
        'contiguous': QtGui.QAbstractItemView.ContiguousSelection,
        'extended': QtGui.QAbstractItemView.ExtendedSelection,
        }
    
    def __init__(self,slist=[],title='Selection Dialog',mode=None,sort=False,func=None,width=None,height=None):
        """Create the SelectionList dialog.
        """
        QtGui.QDialog.__init__(self)
        self.setWindowTitle(title)
        # Selection List
        self.listw = QtGui.QListWidget()
        if width is not None:
            self.listw.setMaximumWidth(width)
        if height is not None:
            self.listw.setMaximumHeight(height)
        self.listw.addItems(slist)
        if sort:
            self.listw.sortItems()
        self.listw.setSelectionMode(self.selection_mode[mode])
        grid = QtGui.QGridLayout()
        grid.addWidget(self.listw,0,0,1,1)
        self.setLayout(grid)
        if func:
            self.connect(self.listw,QtCore.SIGNAL("itemClicked(QListWidgetItem *)"),func)
    

    def setSelected(self,selected,bool):
        """Mark the specified items as selected."""
        for s in selected:
            for i in self.listw.findItems(s,QtCore.Qt.MatchExactly):
                # OBSOLETE: should be changed with Qt version 4.2 or later
                self.listw.setItemSelected(i,bool)
                # SHOULD BECOME:
                # i.setSelected(True) # requires Qt 4.2
                # i.setCheckState(QtCore.Qt.Checked)

                
    def getResult(self):
        """Return the list of selected values.

        If the user cancels the selection operation, the return value is None.
        Else, the result is always a list, possibly empty or with a single
        value.
        """
        res = [i.text() for i in self.listw.selectedItems()]
        return map(str,res)


class Selection(QtGui.QDialog):
    """A dialog for selecting one or more items from a list."""
    
    selection_mode = {
        None: QtGui.QAbstractItemView.NoSelection,
        'single': QtGui.QAbstractItemView.SingleSelection,
        'multi': QtGui.QAbstractItemView.MultiSelection,
        'contiguous': QtGui.QAbstractItemView.ContiguousSelection,
        'extended': QtGui.QAbstractItemView.ExtendedSelection,
        }
    
    def __init__(self,slist=[],title='Selection Dialog',mode=None,sort=False,selected=[]):
        """Create the SelectionList dialog.

        selected is a list of items that are initially selected.
        """
        QtGui.QDialog.__init__(self)
        self.setWindowTitle(title)
        # Selection List
        self.listw = QtGui.QListWidget()
        self.listw.addItems(slist)
        self.listw.setSelectionMode(self.selection_mode[mode])
        if sort:
            self.listw.sortItems()
        if selected:
            self.setSelected(selected)
        # Accept/Cancel Buttons
        acceptButton = QtGui.QPushButton('OK')
        self.connect(acceptButton,QtCore.SIGNAL("clicked()"),self,Accept)
        cancelButton = QtGui.QPushButton('Cancel')
        self.connect(cancelButton,QtCore.SIGNAL("clicked()"),self,Reject)
        # Putting it all together
        grid = QtGui.QGridLayout()
        grid.setColumnStretch(1,1)
        grid.setColumnMinimumWidth(1,250)
        grid.addWidget(self.listw,0,0,1,-1)
        grid.addWidget(acceptButton,1,0)
        grid.addWidget(cancelButton,1,1)
        self.setLayout(grid)
    

    def setSelected(self,selected):
        """Mark the specified items as selected."""
        for s in selected:
            for i in self.listw.findItems(s,QtCore.Qt.MatchExactly):
                # OBSOLETE: should be changed with Qt version 4.2 or later
                # self.listw.setItemSelected(i,True)
                # SHOULD BECOME:
                i.setSelected(True) # requires Qt 4.2
                i.setCheckState(QtCore.Qt.Checked)

                
    def getResult(self):
        """Return the list of selected values.

        If the user cancels the selection operation, the return value is None.
        Else, the result is always a list, possibly empty or with a single
        value.
        """
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            res = [ i.text() for i in self.listw.selectedItems() ]
            return map(str,res)
        else:
            return None
        

# !! The QtGui.QColorDialog can not be instantiated or subclassed.
# !! The color selection dialog is created by the static getColor
# !! function.

def getColor(col=None,caption=None):
    """Create a color selection dialog and return the selected color.

    col is the initial selection.
    If a valid color is selected, its string name is returned, usually as
    a hex #RRGGBB string. If the dialog is canceled, None is returned.
    """
    if type(col) == tuple:
        col = QtGui.QColor.fromRgb(*col)
    else:
        col = QtGui.QColor(col)
    dia = QtGui.QColorDialog
    #myButton = QtGui.QPushButton('MY')
    #dia.layout()
    col = dia.getColor(col)
    if col.isValid():
        return str(col.name())
    else:
        return None

#####################################################################
########### General Input Dialog ####################################
#####################################################################

class InputItem(QtGui.QHBoxLayout):
    """A single input item, usually with a label in front.

    The created widget is a QHBoxLayout which can be embedded in the vertical
    layout of a dialog.
    
    This is a super class for all input items. It just creates a label.
    The input field(s) should be added by the dedicated subclass.

    The constructor has one required argument: ``name``. It is the name used
    to identify the item and should be unique for all InputItems in the same
    dialog. 
    Other (optional) positional parameters are passed to the QHBoxLayout
    constructor.

    By default the constructor adds a label to the QHBoxLayout, with text set
    by the ``text`` keyword argument or (by default) by the name of the item.
    Use the ``text`` argument if you want the displayed text to be different
    from the items name.
    A ``text=''`` parameter will suppress the label. This is e.g. used in the
    InputBoolean constructor, where the text is displayed by the input field.

    The superclass also defines default values for the text(), value() and
    setValue() methods.

    Subclasses should initialize the superclass as follows:
    ``InputItem.__init__(self,name,*args,**kargs)``

    Subclasses should override:
    
    - text(): if they called the superclass __init__() method with ``text=''``;
    - value(): if they did not create a self.input widget who's text() is
      the return value of the item.
    - setValue(): always, unless the field is readonly.

    Subclasses can set validators on the input, like
    ``input.setValidator(QtGui.QIntValidator(input))``
    
    Subclasses can define a show() method e.g. to select the data in the
    input field on display of the dialog.
    """
    
    def __init__(self,name,*args,**kargs):
        """Creates a horizontal box layout and puts the label in it."""
        QtGui.QHBoxLayout.__init__(self,*args)
        self.key = str(name)
        if 'text' in kargs:
            text = kargs['text']
        else:
            text = self.key
        if text:
            self.label = QtGui.QLabel(text)
            self.addWidget(self.label)

        if 'buttons' in kargs and 'parent' in kargs:
            self.buttons = dialogButtons(kargs['parent'],kargs['buttons'])
            self.addLayout(self.buttons)
            

    def name(self):
        """Return the name of the InputItem."""
        return self.key

    def text(self):
        """Return the displayed text of the InputItem."""
        if hasattr(self,'label'):
            return str(self.label.text())
        else:
            return self.key

    def value(self):
        """Return the widget's value."""
        return str(self.input.text())

    def setValue(self,val):
        """Change the widget's value."""
        pass


class InputInfo(InputItem):
    """An unchangeable input field with a label in front.

    It is just like an InputString, but the text can not be edited.
    
    There are no specific options.
    """
    def __init__(self,name,value,*args,**kargs):
        """Creates the input item."""
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QLineEdit(str(value))
        self.input.setReadOnly(True)
        self._value_ = value
        self.insertWidget(1,self.input)

    def value(self):
        """Return the widget's value."""
        return self._value_

    def setValue(self,val):
        """Change the widget's value."""
        self.input.setText(str(val))


class InputString(InputItem):
    """A string input field with a label in front.

    If the type of value is not a string, the input string
    will be eval'ed before returning.
    
    Options:

    - `max`: the maximum number of characters in the string.
    """
    def __init__(self,name,value,max=None,*args,**kargs):
        """Creates the input item."""
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QLineEdit(str(value))
        if max>0:
            self.input.setMaxLength(max)
        self._is_string_ = type(value) == str
        self.insertWidget(1,self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        s = str(self.input.text())
        if self._is_string_:
            return s
        else:
            return eval(s)

    def setValue(self,val):
        """Change the widget's value."""
        self.input.setText(str(val))


class InputText(InputItem):
    """A scrollable text input field with a label in front.
    
    If the type of value is not a string, the input text
    will be eval'ed before returning.
    
    There are no specific op[tions.
    """
    def __init__(self,name,value,*args,**kargs):
        """Creates the input item."""
        InputItem.__init__(self,name,*args,**kargs)
        self._is_string_ = type(value) == str
        self.input =  QtGui.QTextEdit()
        self.setValue(value)
        self.insertWidget(1,self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        s = str(self.input.toPlainText())
        if self._is_string_:
            return s
        else:
            return eval(s)

    def setValue(self,val):
        """Change the widget's value."""
        self.input.setPlainText(str(val))


class InputBool(InputItem):
    """A boolean input item."""
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new checkbox for the input of a boolean value.
        
        Displays the name next to a checkbox, which will initially be set
        if value evaluates to True. (Does not use the label)
        The value is either True or False,depending on the setting
        of the checkbox.
        """
        if 'text' in kargs:
            text = kargs['text']
        else:
            text = str(name)
        kargs['text'] = '' # Force no label
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QCheckBox(text)
        self.setValue(value)
        self.insertWidget(1,self.input)

    def text(self):
        """Return the displayed text."""
        return str(self.input.text())

    def value(self):
        """Return the widget's value."""
        return self.input.checkState() == QtCore.Qt.Checked

    def setValue(self,val):
        """Change the widget's value."""
        if val:
            self.input.setCheckState(QtCore.Qt.Checked)
        else:
            self.input.setCheckState(QtCore.Qt.Unchecked)

    
class InputCombo(InputItem):
    """A combobox InputItem."""
    
    def __init__(self,name,default,choices=[],onselect=None,*args,**kargs):
        """Creates a new combobox for the selection of a value from a list.

        choices is a list/tuple of possible values.
        default is the initial/default choice.
        If default is not in the choices list, it is prepended.
        If default is None, the first item of choices is taken as the default.
       
        The choices are presented to the user as a combobox, which will
        initially be set to the default value.

        An optional `onselect` function may be specified, which will be called
        whenever the current selection changes.
        """
        if len(choices) == 0:
            raise ValueError,"Selection expected choices!"
        if default is None:
            default = choices[0]
        elif default not in choices:
            choices[0:0] = [ default ]
        self._choices_ = [ str(s) for s in choices ]
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QComboBox()
        self.input.addItems(self._choices_)
        if callable(onselect):
            self.connect(self.input,QtCore.SIGNAL("currentIndexChanged(const QString &)"),onselect)
        self.setValue(default)
        self.insertWidget(1,self.input)

    def value(self):
        """Return the widget's value."""
        return str(self.input.currentText())

    def setValue(self,val):
        """Change the widget's value."""
        val = str(val)
        if val in self._choices_:
            self.input.setCurrentIndex(self._choices_.index(val))

    
class InputRadio(InputItem):
    """A radiobuttons InputItem."""
    
    def __init__(self,name,default,choices=[],direction='h',*args,**kargs):
        """Creates radiobuttons for the selection of a value from a list.

        choices is a list/tuple of possible values.
        default is the initial/default choice.
        If default is not in the choices list, it is prepended.
        If default is None, the first item of choices is taken as the default.
       
        The choices are presented to the user as a hbox with radio buttons,
        of which the default will initially be pressed.
        If direction == 'v', the options are in a vbox. 
        """
        if default is None:
            default = choices[0]
        elif default not in choices:
            choices[0:0] = [ default ]
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QGroupBox()
        if direction == 'v':
            self.hbox = QtGui.QVBoxLayout()
            self.hbox.setContentsMargins(0,10,0,10)
        else:
            self.hbox = QtGui.QHBoxLayout()
            self.hbox.setContentsMargins(10,0,10,0)
        self.rb = []
        self.hbox.addStretch(1)
        
        for v in choices:
            rb = QtGui.QRadioButton(v)
            self.hbox.addWidget(rb)
            self.rb.append(rb)

        self.rb[choices.index(default)].setChecked(True)
        self.input.setLayout(self.hbox)
        self.insertWidget(1,self.input)

    def value(self):
        """Return the widget's value."""
        for rb in self.rb:
            if rb.isChecked():
                return str(rb.text())
        return ''

    def setValue(self,val):
        """Change the widget's value."""
        val = str(val)
        for rb in self.rb:
            if rb.text() == val:
                rb.setChecked(True)
                break

    
class InputPush(InputItem):
    """A pushbuttons InputItem."""
    
    def __init__(self,name,default=None,choices=[],direction='h',*args,**kargs):
        """Creates pushbuttons for the selection of a value from a list.

        choices is a list/tuple of possible values.
        default is the initial/default choice.
        If default is not in the choices list, it is prepended.
        If default is None, the first item of choices is taken as the default.
       
        The choices are presented to the user as a hbox with radio buttons,
        of which the default will initially be selected.
        If direction == 'v', the options are in a vbox. 
        """
        if default is None:
            default = choices[0]
        elif default not in choices:
            choices[0:0] = [ default ]
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QGroupBox()
        self.input.setFlat(True)
        if direction == 'v':
            self.hbox = QtGui.QVBoxLayout()
            self.hbox.setContentsMargins(0,10,0,10)
        else:
            self.hbox = QtGui.QHBoxLayout()
            self.hbox.setContentsMargins(10,5,10,5)
        self.hbox.setSpacing(0)
        self.hbox.setMargin(0)

        self.rb = []
        for v in choices:
            rb = QtGui.QPushButton(v)
            self.hbox.addWidget(rb)
            self.rb.append(rb)

        self.rb[choices.index(default)].setChecked(True)
        self.input.setLayout(self.hbox)
        self.insertWidget(1,self.input)

    def setText(self,text,index=0):
        """Change the text on button index."""
        self.rb[index].setText(text)

    def setIcon(self,icon,index=0):
        """Change the icon on button index."""
        self.rb[index].setIcon(icon)

    def value(self):
        """Return the widget's value."""
        for rb in self.rb:
            if rb.isChecked():
                return str(rb.text())
        return ''

    def setValue(self,val):
        """Change the widget's value."""
        val = str(val)
        for rb in self.rb:
            if rb.text() == val:
                rb.setChecked(True)
                break


class InputInteger(InputItem):
    """An integer input item.

    Options:

    - 'min, 'max': range of the scale (integer)
    """
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new integer input field with a label in front."""
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QLineEdit(str(value))
        self.validator = QtGui.QIntValidator(self)
        if kargs.has_key('min'):
            self.validator.setBottom(int(kargs['min']))
        if kargs.has_key('max'):
            self.validator.setTop(int(kargs['max']))
        self.input.setValidator(self.validator)
        self.insertWidget(1,self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        return int(self.input.text())

    def setValue(self,val):
        """Change the widget's value."""
        val = int(val)
        self.input.setText(str(val))


class InputFloat(InputItem):
    """An float input item."""
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new float input field with a label in front."""
        InputItem.__init__(self,name,*args,**kargs)
        self.input = QtGui.QLineEdit(str(value))
        self.validator = QtGui.QDoubleValidator(self)
        if kargs.has_key('min'):
            self.validator.setBottom(float(kargs['min']))
        if kargs.has_key('max'):
            self.validator.setTop(float(kargs['max']))
        if kargs.has_key('dec'):
            self.validator.setDecimals(int(kargs['dec']))
        self.input.setValidator(self.validator)
        self.insertWidget(1,self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        return float(self.input.text())

    def setValue(self,val):
        """Change the widget's value."""
        val = float(val)
        self.input.setText(str(val))

   
class InputSlider(InputInteger):
    """An integer input item using a slider.

    Options:
    
    - ``min``, ``max``: range of the scale (integer)
    - ``ticks`` : step for the tick marks (default range length / 10)
    - ``func`` : an optional function to be called whenever the value is
      changed. The function takes a float/integer argument.
    """
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new integer input slider."""
        InputInteger.__init__(self,name,value,*args,**kargs)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        vmin = kargs.get('min',0)
        vmax = kargs.get('max',100)
        
        ticks = kargs.get('ticks',(vmax-vmin)/10)
        self.slider.setTickInterval(ticks)
        self.slider.setMinimum(vmin)
        self.slider.setMaximum(vmax)
        self.slider.setValue(value)
        self.slider.setSingleStep(1)
        #self.slider.setPageStep(5)
        self.slider.setTracking(1)
        self.connect(self.slider,QtCore.SIGNAL("valueChanged(int)"),self.set_value)
        if kargs.has_key('func'):
            self.connect(self.slider,QtCore.SIGNAL("valueChanged(int)"),kargs['func'])            
        self.addWidget(self.slider)

    def set_value(self,val):
        val = int(val)
        self.input.setText(str(val))

   
class InputFSlider(InputFloat):
    """A float input item using a slider.

    Options:
    
    - 'min', 'max': range of the scale (integer)
    - 'scale': scale factor to compute the float value
    - 'ticks' : step for the tick marks (default range length / 10)
    - 'func' : an optional function to be called whenever the value is
      changed. The function takes a float/integer argument.
    """
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new integer input slider."""
        InputFloat.__init__(self,name,value,*args,**kargs)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.scale = kargs.get('scale',1.0)
        self.func = kargs.get('func',None)

        vmin = kargs.get('min',0)
        vmax = kargs.get('max',100)
        ticks = kargs.get('ticks',(vmax-vmin)/10)
        self.slider.setTickInterval(ticks)
        self.slider.setMinimum(vmin)
        self.slider.setMaximum(vmax)
        self.slider.setValue(value/self.scale)
        self.slider.setSingleStep(1)
        #self.slider.setPageStep(5)
        self.slider.setTracking(1)
        self.connect(self.slider,QtCore.SIGNAL("valueChanged(int)"),self.set_value)
        self.addWidget(self.slider)

    def set_value(self,val):
        val = float(val)
        value = val*self.scale
        self.input.setText(str(value))
        if self.func:
            self.func(value)


class InputColor(InputItem):
    """A color input item."""
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new color input field with a label in front.

        The color input field is a button displaying the current color.
        Clicking on the button opens a color dialog, and the returned
        value is set in the button.
        """
        InputItem.__init__(self,name,*args,**kargs)
        color = colors.colorName(value)
        self.input = QtGui.QPushButton(color)
        self.setValue(color)
        self.connect(self.input,QtCore.SIGNAL("clicked()"),self.setColor)
        self.insertWidget(1,self.input)
        
    def setColor(self):
        color = getColor(self.input.text())
        if color:
            self.setValue(color)

    def setValue(self,value):
        """Change the widget's value."""
        rgb = QtGui.QColor(value).getRgb()
        self.input.setStyleSheet("* { background-color: rgb(%s,%s,%s) }" % rgb[:3])
        self.input.setText(str(value))


def inputAny(name,value,itemtype=str,**options):
    """Create an InputItem of any type, depending on the arguments.

    Arguments: only name and value are required

    - name: name of the item, also the key for the return value
    - value: initial value,
    - itemtype: one of the available itemtypes, default derived from value or
      str if value is not recognized.
    - text: descriptive text displayed on the input dialog, default equal to
      name
    - choices: a list of posible values (for selection types)
    - min,max: limits for range types
    - validator: customized validation function
    """
    if itemtype == bool:
        line = InputBool(name,value,**options)

    elif itemtype == int:
        line = InputInteger(name,value,**options)

    elif itemtype == float:
        line = InputFloat(name,value,**options)

    elif itemtype == 'slider':
        if type(value) == int:
            line = InputSlider(name,value,**options)
            
        elif type(value) == float:
            line = InputFSlider(name,value,**options)

    elif itemtype == 'info':
        line = InputInfo(name,value,**options)

    elif itemtype == 'text':
        line = InputText(name,value,**options)

    elif itemtype == 'color':
        line = InputColor(name,value,**options)

    elif itemtype == 'select' :
        line = InputCombo(name,value,**options)

    elif itemtype in ['radio','hradio','vradio']:
        options['direction'] = itemtype[0]
        line = InputRadio(name,value,**options)

    elif itemtype in ['push','hpush','vpush']:
        options['direction'] = itemtype[0]
        line = InputPush(name,value,**options)

    else: # Anything else is handled as a string
        #itemtype = str:
        line = InputString(name,value,**options)
    return line
                

def inputAnyOld(item,parent=None):
    """Create an InputItem with the old data style.

    This translates the data from the legacy InputItem data to the
    new style required by InputAny.
    Returns the InputItem constrctured with the data.
    """
    name,value = item[:2]
    
    if type(item[-1]) == dict:
        # we have options
        options = item[-1]
        item = item[:-1]
    else:
        options = {}

    if len(item) > 2 and type(item[2]) == str:
        itemtype = item[2]
    else:
        # No item specified: guess from value or from available options
        if 'choices' in options:
            itemtype = 'select'
        else:
            itemtype = type(value)

    if itemtype == int:
        if len(item) > 3 and type(item[3] != dict):
            options['min'] = int(item[3])
        if len(item) > 4:
            options['max'] = int(item[4])

    elif itemtype == float:
        if len(item) > 3 and type(item[3] != dict):
            options['min'] = int(item[3])
        if len(item) > 4:
            options['max'] = int(item[4])
        if len(item) > 5:
            options['dec'] = int(item[5])

    elif itemtype == 'select' :
        if len(item) > 3:
            options['choices'] = item[3]

    elif itemtype in ['radio','hradio','vradio']:
        if len(item) > 3:
            options['choices'] = item[3]
        options['direction'] = itemtype[0]

    elif itemtype in ['push','hpush','vpush']:
        if len(item) > 3:
            options['choices'] = item[3]
        options['direction'] = itemtype[0]

    if parent is not None:
        options['parent'] = parent

    return inputAny(name,value,itemtype,**options)


class InputDialog(QtGui.QDialog):
    """A dialog widget to set the value of one or more items.

    While general input dialogs can be constructed from all the underlying
    Qt classes, this widget provides a way to construct fairly complex
    input dialogs with a minimum of effort.

    The input dialog can be modal or non-modal dialog.
    """
    
    def __init__(self,items,caption=None,parent=None,flags=None,actions=None,default=None,scroll=False):
        """Create a dialog asking the user for the value of items.

        `items` is either a list of items, or a dict where each value is a
        list of items. If `items` is a dict, a tabbed widget will be created
        with a tab for each (key,value) pair in the dict.

        Each item in an `items` list is a list or tuple of the form
        (name,value,type,options), where the fields have the following meaning:
    
        - name:  the name of the field,
        - value: the initial or default value of the field,
        - type:  the type of values the field can accept,
        - options: a dict with options for the field.

        At least the name and initial value need to be specified. The type
        can often be determined from the initial value. Some types set the
        initial value from an option if it was an empty string or None.
        The options dictionary has both generic options, available for all
        item types, and type specific options.

        Each item specifies a single input field, and its value will be
        contained in the results dictionary using the field name as a key.
        
        For each item a single input line is created in the dialog.
        This line by default consists of a label displaying the field
        name and a LineEdit widget where the initial value is displayed
        and can be changed. Where appropriate, a validator function is attached
        to it.

        The following options are applicable to all item types:

        - text: if specified, the text value will be displayed instead of
          the name. The name value will remain the key in the return dict.
          Use this field to display a more descriptive text for the user,
          while using a short name for handling the value in your script.
        - buttons:
        - tooltip:

        Currently, the following item types are available:

        The item specific options:
        - min
        - max
        - range: the range of values the field can accept,
        - choices

        The first two fields are mandatory. In many cases the type can be
        determined from the value and no other fields are required. Thus:

        - [ 'name', 'value' ] will accept any string (initial string = 'value'),
        - [ 'name', True ] will show a checkbox with the item checked,
        - [ 'name', 10 ] will accept any integer,
        - [ 'name', 1.5 ] will accept any float.

        Range settings for int and float types:

        - [ 'name', 1, int, 0, 4 ] will accept an integer from 0 to 4, inclusive
        - [ 'name', 1, float, 0.0, 1.0, 2 ] will accept a float in the range
          from 0.0 to 1.0 with a maximum of two decimals.

        Composed types:

        - [ 'name', 'option1', 'select', ['option0','option1','option2']] will
          present a combobox to select between one of the options.
          The initial and default value is 'option1'.
        - [ 'name', 'option1', 'radio', ['option0','option1','option2']] will
          present a group of radiobuttons to select between one of the options.
          The initial and default value is 'option1'.
          A variant 'vradio' aligns the options vertically. 
        - [ 'name', 'option1', 'push', ['option0','option1','option2']] will
          present a group of pushbuttons to select between one of the options.
          The initial and default value is 'option1'.
          A variant 'vpush' aligns the options vertically. 
        - [ 'name', 'red', 'color' ] will present a color selection widget,
          with 'red' as the initial choice.
        """
        if parent is None:
            parent = GD.GUI
        QtGui.QDialog.__init__(self,parent)
        if flags is not None:
            self.setWindowFlags(flags)
        if caption is None:
            caption = 'pyFormex-dialog'
        self.setWindowTitle(str(caption))
        self.fields = []
        self.results = {}
        self._pos = None
        form = QtGui.QVBoxLayout()

        if isinstance(items,dict):
            # add the input tab pages
            tab = QtGui.QTabWidget()
            for page in items:
                w = QtGui.QWidget()
                f = QtGui.QVBoxLayout()
                # add the items to the tab page
                for item in items[page]:
                    line = inputAnyOld(item,parent=self)
                    f.addLayout(line)
                    self.fields.append(line)
                f.addStretch()
                w.setLayout(f)
                tab.addTab(w,page)
            form.addWidget(tab)

        else:
            # add the items directly
            for item in items:
                line = inputAnyOld(item,parent=self)
                form.addLayout(line)
                self.fields.append(line)

        # add the action buttons
        if actions is None:
            actions = [('CANCEL',),('OK',)]
            default = 'OK'
        but = dialogButtons(self,actions,default)
        form.addLayout(but)
        if scroll:
            # This is experimental !!!
            self.child = QtGui.QWidget()
            self.child.setLayout(form)
            self.scroll = QtGui.QScrollArea(self)
            self.scroll.setWidget(self.child)
            self.scroll.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
            self.scroll.resize(GD.GUI.width()/2,GD.GUI.height())
        else:
            self.setLayout(form)
        self.connect(self,QtCore.SIGNAL("accepted()"),self.acceptData)
        

    def __getitem__(self,name):
        """Return the input item with specified name."""
        items = [ f for f in self.fields if f.name() == name ]
        if len(items) > 0:
            return items[0]
        else:
            return None


    def timeout(self):
        """Hide the dialog and set the result code to TIMEOUT"""
        GD.debug("TIMEOUT")
        self.acceptData(TIMEOUT)


    def timedOut(self):
        """Returns True if the result code was set to TIMEOUT"""
        return self.result() == TIMEOUT


    def show(self,timeout=None,timeoutfunc=None,modal=False):
        """Show the dialog.

        For a non-modal dialog, the user has to call this function to
        display the dialog. 
        For a modal dialog, this is implicitely executed by getResult().

        If a timeout is given, start the timeout timer.
        """
        # Set the keyboard focus to the first input field
        self.fields[0].input.setFocus()
        self.status = None

        self.setModal(modal)
        QtGui.QDialog.show(self)

        addTimeOut(self,timeout,timeoutfunc)
        
        
    def acceptData(self,result=ACCEPTED):
        """Update the dialog's return value from the field values.

        This function is connected to the 'accepted()' signal.
        Modal dialogs should normally not need to call it.
        In non-modal dialogs however, you can call it to update the
        results without having to raise the accepted() signal (which
        would close the dialog).
        """
        #GD.debug("ACCEPTING DATA WITH RESULT %s"%result)
        self.results = {}
        self.results.update([ (fld.name(),fld.value()) for fld in self.fields ])
        ## if self.report_pos:
        ##     self.results.update({'__pos__':self.pos()})
        if result == TIMEOUT:
            self.done(result)
        else:
            self.setResult(result)
        

    def updateData(self,d):
        """Update a dialog from the data in given dictionary.

        d is a dictionary where the keys are field names in the dialog.
        The values will be set in the corresponding input items.
        """
        for f in self.fields:
            n = f.name()
            if n in d:
                f.setValue(d[n])
        
        
    def getResult(self,timeout=None):
        """ Get the results from the input dialog.

        This fuction is used to present a modal dialog to the user (i.e. a
        dialog that must be ended before the user can continue with the
        program. The dialog is shown and user interaction is processed.
        The user ends the interaction either by accepting the data (e.g. by
        pressing the OK button or the ENTER key) or by rejecting them (CANCEL
        button or ESC key).
        On accept, a dictionary with all the fields and their values is
        returned. On reject, an empty dictionary is returned.
        
        If a timeout (in seconds) is given, a timer will be started and if no
        user input is detected during this period, the input dialog returns
        with the default values set.
        A value 0 will timeout immediately, a negative value will never timeout.
        The default is to use the global variable input_timeout.

        The result() method can be used to find out how the dialog was ended.
        Its value will be one of ACCEPTED, REJECTED ot TIMEOUT.
        """
        self.results = {}
        self.setResult(0)
        if self._pos is not None:
            self.restoreGeometry(self._pos)
            
        self.show(timeout,modal=True)
        self.exec_()
        self.activateWindow()
        self.raise_()
        GD.app.processEvents()
        self._pos = self.saveGeometry()
        return self.results

    
def updateDialogItems(data,newdata):
    """Update the input data fields with new data values

    data: a list of dialog items, as required by an InputDialog.
    newdata: a dictionary with new values for (some of) the items.
    The values in data which have a matching key in newdata will be
    replaced with the new value, unless it is None.
    
    This function requires that the dialog items are lists, not tuples.
    """
    if newdata:
        if type(data) is dict:
            for d in data:
                #print("subitems",d)
                updateDialogItems(data[d],newdata)
        else:
            for d in data:
                #print("item",d)
                v = newdata.get(d[0],None)
                #print(v)
                if v is not None:
                    d[1] = v
                #print(d)


def dialogButtons(dialog,actions,default=None):
    """Create a set of dialog buttons

    dia is a dialog widget
    actions is a list of tuples (name,) or (name,function).
    If a function is specified, it will be executed on pressing the button.
    If no function is specified, and name is one of 'ok' or 'cancel' (case
    does not matter), the button will be bound to the dialog's 'accept'
    or 'reject' slot.
    default is the name of the action to set as the default.
    """
    but = QtGui.QHBoxLayout()
    spacer = QtGui.QSpacerItem(0,0,QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum )
    but.addItem(spacer)
    for a in actions:
        name = a[0]
        b = QtGui.QPushButton(name,dialog)
        n = name.lower()
        if len(a) > 1:
            slot = (a[1],)
        elif n == 'ok':
            slot = (dialog,Accept)
        elif n == 'cancel':
            slot = (dialog,Reject)
        else:
            slot = (dialog,Reject)
        dialog.connect(b,QtCore.SIGNAL("clicked()"),*slot)
        if default is not None and n == default.lower():
            b.setDefault(True)
        but.addWidget(b)
    return but


########################### Table widgets ###########################


class TableModel(QtCore.QAbstractTableModel):
    """A table model that represent data as a two-dimensional array of items.

    data is any tabular data organized in a fixed number of rows and colums.
    This means that an item at row i and column j can be addressed as
    data[i][j].
    Optional lists of column and row headers can be specified.
    """
    def __init__(self,data,chead=None,rhead=None,parent=None,*args): 
        QtCore.QAbstractTableModel.__init__(self,parent,*args) 
        self.arraydata = data
        self.headerdata = {QtCore.Qt.Horizontal:chead,QtCore.Qt.Vertical:rhead}
 
    def rowCount(self,parent=None): 
        return len(self.arraydata) 
 
    def columnCount(self,parent=None): 
        return len(self.arraydata[0]) 
 
    def data(self,index,role): 
        if index.isValid() and role == QtCore.Qt.DisplayRole: 
            return QtCore.QVariant(self.arraydata[index.row()][index.column()]) 
        return QtCore.QVariant() 

    def headerData(self,col,orientation,role):
        if orientation in self.headerdata and self.headerdata[orientation] and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.headerdata[orientation][col])
        return QtCore.QVariant()

    def insertRows(self,row=None,count=None):
        if row is None:
            row = self.rowCount()
        if count is None:
            count = 1
        last = row+count-1
        newdata = [ [ None ] * self.columnCount() ] * count
        self.beginInsertRows(QtCore.QModelIndex(),row,last)
        self.arraydata[row:row] = newdata
        self.endInsertRows()
        return True

    def removeRows(self,row=None,count=None):
        if row is None:
            row = self.rowCount()
        if count is None:
            count = 1
        last = row+count-1
        self.beginRemoveRows(QtCore.QModelIndex(),row,last)
        self.arraydata[row:row+count] = []
        self.endRemoveRows()
        return True


class Table(QtGui.QDialog):
    """A dialog widget to show two-dimensional arrays of items."""
    
    def __init__(self,data,chead=None,rhead=None,caption=None,parent=None,actions=[('OK',)],default='OK'):
        """Create the Table dialog.
        
        data is a 2-D array of items, mith nrow rows and ncol columns.
        chead is an optional list of ncol column headers.
        rhead is an optional list of nrow row headers.
        """
        if parent is None:
            parent = GD.GUI
        QtGui.QDialog.__init__(self,parent)
        if caption is None:
            caption = 'pyFormex-dialog'
        self.setWindowTitle(str(caption))
        
        form = QtGui.QVBoxLayout()
        table = QtGui.QTableView()
        tm = TableModel(data,chead,rhead,None)
        table.setModel(tm)
        table.horizontalHeader().setVisible(chead is not None)
        table.verticalHeader().setVisible(rhead is not None)
        table.resizeColumnsToContents()
        form.addWidget(table)

        but = dialogButtons(self,actions,default)
        form.addLayout(but)
        
        self.setLayout(form)
        #self.setMinimumSize(1000,400)
        self.table = table
        self.show()


class TableDialog(QtGui.QDialog):
    """A dialog widget to show two-dimensional arrays of items."""
    def __init__(self,items,caption=None,parent=None,tab=False):
        """Create the Table dialog.
        
        If tab = False, a dialog with one table is created and items
        should be a list [table_header,table_data].
        If tab = True, a dialog with multiple pages is created and items
        should be a list of pages [page_header,table_header,table_data].
        """
        if parent is None:
            parent = GD.GUI
        QtGui.QDialog.__init__(self,parent)
        if caption is None:
            caption = 'pyFormex-table'
        self.setWindowTitle(str(caption))
        form = QtGui.QVBoxLayout()
        if tab:
            tables = QtGui.QTabWidget()
            for item in items:
                page = QtGui.QTableView()
                page_header,table_header,table_data = item
                tm = TableModel(table_data,table_header,parent=self)
                page.setModel(tm)
                page.verticalHeader().setVisible(False)
                page.resizeColumnsToContents()
                tables.addTab(page,page_header)
            form.addWidget(tables)
        else:
            table = QtGui.QTableView()
            table_header,table_data = items
            tm = TableModel(table_data,table_header,parent=self)
            table.setModel(tm)
            table.verticalHeader().setVisible(False)
            table.resizeColumnsToContents()
            form.addWidget(table)
        self.setLayout(form)
        self.setMinimumSize(1000,400)
        self.show()


#####################################################################
########### Text Display Widgets ####################################
#####################################################################


def updateText(widget,text,format=''):
    """Update the text of a text display widget.

    - `widget`: a widget that has the :meth:`setText`, :meth:`setPlainText`
      and :meth:`setHtml` methods to set the widget's text.
      Examples are :class:`QMessageBox` and :class:`QTextEdit`.
    - `text`: a multiline string with the text to be displayed.
    - `format`: the format of the text. If empty, autorecognition will be
      tried. Currently available formats are: ``plain``, ``html`` and
      ``rest`` (reStructuredText).

    This function allows to display other text formats besides the
    plain text and html supported by the widget. 
    Any format other than ``plain`` or ``html`` will be converted to one
    of these before sending it to the widget. Currently, we convert the
    following formats:
    
    ``rest`` (reStructuredText): 
      If the :mod:docutils is available, `rest` text is converted to `html`,
      otherwise it will be displayed as plain text.
      A text is autorecognized as reStructuredText if its first
      line starts with '..'. Note: If you add a '..' line to your text to
      have it autorecogn ized as reST, be sure to have it followed with a
      blank line, or your first paragraph could be turned into comments.
      
    """
    # autorecognition
    if format not in ['plain','html','rest']:
        if type(text) is str and text.startswith('..'):
            format = 'rest'

    # conversion
    if format == 'rest':
        try:
            text = utils.rst2html(text)
        except:
            GD.message("Could not convert reStrucuturedText to html. Displaying as plain text.")
            if GD.options.debug:
                raise
            
        format = ''
        # We leave the format undefined, because we are not sure
        # that the conversion function (docutils) is available
        
    if format == 'plain':
        widget.setPlainText(text)
    elif format == 'html':
        widget.setHtml(text)
    else:
        # As a last rescue, try QT4's autorecognition
        widget.setText(text)


class MessageBox(QtGui.QMessageBox):
    """A message box is a widget displaying a short text for the user.

    The message box displays a text, an optional icon depending on the level
    and a number of push buttons.

    - `text`: the text to be shown. This can be either plain text or html
      or reStructuredText.
    - `format`: the text format: either 'plain', 'html' or 'rest'.
      Any other value will try automatic recognition.
      Recognition of plain text and html is automatic.
      A text is autorecognized to be reStructuredText if its first
      line starts with '..' and is followed by a blank line.
    - `level`: defines the icon that will be shown together with the text.
      If one of 'question', 'info', 'warning' or 'error', a matching icon
      will be shown to hint the user about the type of message. Any other
      value will suppress the icon.
    - `actions`: a list of strings. For each string a pushbutton will be
      created which can be used to exit the dialog and remove the message.
      By default there is a single button labeled 'OK'.

    When the MessageBox is displayed with the :meth:`getResult()` method,
    a modal
    dialog is created, i.e. the user will have to click a button or hit the
    ESC key before he can continue.

    If you want a modeless dialog, allowing the user to continue while the
    message stays open, use the :meth:`show()` mehod to display it.
    """
    def __init__(self,text,format='',level='info',actions=['OK'],default=None,timeout=None):
        QtGui.QMessageBox.__init__(self)
        if default is None:
            default = actions[-1]
        updateText(self,text,format)
        if level == 'error':
            self.setIcon(QtGui.QMessageBox.Critical)
        elif level == 'warning':
            self.setIcon(QtGui.QMessageBox.Warning)
        elif level == 'info':
            self.setIcon(QtGui.QMessageBox.Information)
        elif level == 'question':
            self.setIcon(QtGui.QMessageBox.Question)
        for a in actions:
            b = self.addButton(a,QtGui.QMessageBox.AcceptRole)
            if a == default:
                self.setDefaultButton(b)

        addTimeOut(self,timeout,"accept()")


    def getResult(self):
        """Display the message box and wait for user to click a button.

        This will show the message box as a modal dialog, so that the
        user has to click a button (or hit the ESC key) before he can continue.
        Returns the text of the button that was clicked or
        an empty string if ESC was hit.
        """
        self.exec_()
        b = self.clickedButton()
        if not b:  # b == 0 or b is None
            b = self.defaultButton()
        if b:
            return str(b.text())
        else:
            return ''


class TextBox(QtGui.QDialog):
    """Display a text and wait for user response.

    Possible choices are 'OK' and 'CANCEL'.
    The function returns True if the OK button was clicked or 'ENTER'
    was pressed, False if the 'CANCEL' button was pressed or ESC was pressed.
    """
    def __init__(self,text,format=None,actions=['OK',None]):
        QtGui.QDialog.__init__(self)
        self.setWindowTitle('pyFormex Text Display')
        self._t = QtGui.QTextEdit()
        self._t.setReadOnly(True)
        updateText(self._t,text,format)
        self._b = ButtonBox('',actions,parent=self,stretch=[1,1]) 
        l = QtGui.QVBoxLayout()
        l.addWidget(self._t)
        l.addWidget(self._b)
        self.setLayout(l)
        self.resize(800,400)

    def getResult(self):
        return self.exec_() == QtGui.QDialog.Accepted


############################# Button box ###########################

class ButtonBox(QtGui.QWidget):
    """A box with action buttons.

    - `name`: a label to be displayed in front of the button box. An empty
      string will suppress it.
    - `actions`: a list of (button label, button function) tuples. The button
      function can be a normal callable function, or one of the values
      widgets.ACCEPTED or widgets.REJECTED. In the latter case, `parent`
      should be specified.
    - `parent`: the parent dialog holding this button box. It should be
      specified if one of the buttons actions is widgets.ACCEPTED or
      widgets.REJECTED.
    """
    def __init__(self,name,actions=[],parent=None,stretch=[-1,-1]):
        QtGui.QWidget.__init__(self,parent)
        s = InputPush(name,None,[a[0] for a in actions])
        for i in [0,-1]:
            if stretch[i] >= 0:
                s.insertStretch(i,stretch[i])
        s.setSpacing(0)
        s.setMargin(0)
        for r,f in zip(s.rb,[a[1] for a in actions]):
            if f is None:
                f = Accept
            if callable(f):
                self.connect(r,QtCore.SIGNAL("clicked()"),f)
            else:
                self.connect(r,QtCore.SIGNAL("clicked()"),parent,f)
        self.setLayout(s)
        self.buttons = s

    def setText(self,text,index=0):
        self.buttons.setText(text,index)

    def setIcon(self,icon,index=0):
        self.buttons.setIcon(icon,index)

    def __str__(self):
        s = ''
        for a in ['rect','minimumHeight']:
            s += "%s = %s\n" % (a,getattr(self,a)())
        return s


############################# Combo box ###########################

class ComboBox(QtGui.QWidget):
    def __init__(self,name,choices,func,*args):
        QtGui.QWidget.__init__(self,*args)
        s = InputCombo(name,None,choices=choices,*args)
        s.setSpacing(0)
        s.setMargin(0)
        if func:
            self.connect(s.input,QtCore.SIGNAL("activated(int)"),func)
        self.setLayout(s)
        self.combo = s

    def setIndex(self,i):
        self.combo.input.setCurrentIndex(i)


############################# Menu ##############################


def normalize(s):
    """Normalize a string.

    Text normalization removes all '&' characters and converts to lower case.
    """
    return str(s).replace('&','').lower()


class BaseMenu(object):
    """A general menu class.

    A hierarchical menu that keeps a list of its item names and actions.
    The item names are normalized by removing all '&' characters and
    converting the result to lower case.
    It thus becomes easy to search for an existing item in a menu.
    
    This class is not intended for direct use, but through subclasses.
    Subclasses should implement at least the following methods:
    
    - addSeparator()
    - insertSeperator(before)
    - addAction(text,action)
    - insertAction(before,text,action)
    - addMenu(text,menu)
    - insertMenu(before,text,menu)
      
    QtGui.Menu and QtGui.MenuBar provide these methods.
    """

    def __init__(self,title='AMenu',parent=None,before=None,items=None):
        """Create a menu."""
        self.title = title
        GD.debug("Creating menu %s" % title)
        self.parent = parent
        self.menuitems = odict.ODict()
        if items:
            self.insertItems(items)
        if parent and isinstance(parent,BaseMenu):
            if before:
                before = parent.itemAction(before)
            parent.insert_menu(self,before)
            title = normalize(title)
            if not title in parent.menuitems:
                parent.menuitems[title] = self


    def item(self,text):
        """Get the menu item with text.

        The text will be normalized before searching it.
        If an item with the resulting name exists, it is returned.
        Else None is returned.
        """
        return self.menuitems.get(normalize(text),None)


    def itemAction(self,item):
        """Return the action corresponding to item.

        item is either one of the menu's item texts, or one of its
        values. This method guarantees that the return value is either the
        corresponding Action, or None.
        """
        if item not in self.menuitems.values():
            item = self.item(item)
        if isinstance(item,QtGui.QMenu):
            item = item.menuAction()
        return item
    
    # The need for the following functions demonstrates how much more
    # powerful a dynamically typed language as Python is as compared to
    # the C++ language used by Qt
    def insert_sep(self,before=None):
        """Create and insert a separator"""
        if before:
            return self.insertSeparator(before)
        else:
            return self.addSeparator()

    def insert_menu(self,menu,before=None):
        """Insert an existing menu."""
        if before:
            return self.insertMenu(before,menu)
        else:
            return self.addMenu(menu)

    def insert_action(self,action,before=None):
        """Insert an action.""" 
        if before:
            return self.insertAction(before,action)
        else:
            return self.addAction(action)

    def create_insert_action(self,str,val,before=None):
        """Create and insert an action.""" 
        if before:
            return self.insertAction(before,str,val)
        else:
            return self.addAction(str,val)
    

    def insertItems(self,items,before=None):
        """Insert a list of items in the menu.
        
        Each item is a tuple of two to five elements:
        Text, Action, [ Icon,  ShortCut, ToolTip ].

        Item text is the text that will be displayed in the menu.
        It will be stored in a normalized way: all lower case and with
        '&' removed.

        Action can be any of the following:
        
        - a Python function or instance method : it will be called when the
          item is selected,
        - a string with the name of a function/method,
        - a list of Menu Items: a popup Menu will be created that will appear
          when the item is selected,
        - None : this will create a separator item with no action.

        Icon is the name of one of the icons in the installed icondir.
        ShortCut is an optional key combination to select the item.
        Tooltip is a popup help string.

        If before is given, it specifies the text OR the action of one of the
        items in the menu: the new items will be inserted before that one.
        """
        if before:
            before = self.itemAction(before)
        for item in items:
            txt,val = item[:2]
            if  val is None:
                a = self.insert_sep(before)
            elif isinstance(val, list):
                a = Menu(txt,self)
                a.insertItems(val)
            else:
                if type(val) == str:
                    val = eval(val)
                if len(item) > 2 and item[2].has_key('data'):
                    a = DAction(txt,data=item[2]['data'])
                    QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),val)
                    self.insert_action(a,before)
                else:
                    a = self.create_insert_action(txt,val,before)
                if len(item) > 2:
                    for k,v in item[2].items():                        
                        if k == 'icon':
                            a.setIcon(QtGui.QIcon(QtGui.QPixmap(utils.findIcon(v))))
                        elif k == 'shortcut':
                            a.setShortcut(v)
                        elif k == 'tooltip':
                            a.setToolTip(v)
                        elif k == 'checkable':
                            a.setCheckable(v)
                        elif k == 'disabled':
                            a.setDisabled(True)
            txt = normalize(txt)
            if not txt in self.menuitems:
                self.menuitems[txt] = a


class Menu(BaseMenu,QtGui.QMenu):
    """A popup/pulldown menu."""

    def __init__(self,title='UserMenu',parent=None,before=None,items=None):
        """Create a popup/pulldown menu.

        If parent==None, the menu is a standalone popup menu.
        If parent is given, the menu will be inserted in the parent menu.
        If parent==GD.GUI, the menu is inserted in the main menu bar.
        
        If insert == True, the menu will be inserted in the main menubar
        before the item specified by before.
        If before is None or not the normalized text of an item of the
        main menu, the new menu will be inserted at the end.
        Calling the close() function of an inserted menu will remove it
        from the main menu.

        If insert == False, the created menu will be an independent dialog
        and the user will have to process it explicitely.
        """
        QtGui.QMenu.__init__(self,title,parent)
        BaseMenu.__init__(self,title,parent,before,items)
        if parent is None:
            self.setWindowFlags(QtCore.Qt.Dialog)
            self.setWindowTitle(title)
        self.done = False
            

    def process(self):
        if not self.done:
            if not self.insert:
                self.show()
            GD.app.processEvents()


    def remove(self):
        """Remove this menu from its parent."""
        self.done=True
        if self.parent:
            self.parent.removeAction(self.menuAction())
            for k,v in self.parent.menuitems.items():
                if v == self:
                    del self.parent.menuitems[k]


class MenuBar(BaseMenu,QtGui.QMenuBar):
    """A menu bar allowing easy menu creation."""

    def __init__(self,title='TopMenuBar'):
        """Create the menubar."""
        QtGui.QMenuBar.__init__(self)
        BaseMenu.__init__(self,title)


###################### Action List ############################################

class DAction(QtGui.QAction):
    """A DAction is a QAction that emits a signal with a string parameter.

    When triggered, this action sends a signal (default 'Clicked') with a
    custom string as parameter. The connected slot can then act depending
    on this parameter.
    """

    signal = "Clicked"
    
    def __init__(self,name,icon=None,data=None,signal=None):
        """Create a new DAction with name, icon and string data.

        If the DAction is used in a menu, a name is sufficient. For use
        in a toolbar, you will probably want to specify an icon.
        When the action is triggered, the data is sent as a parameter to
        the SLOT function connected with the 'Clicked' signal.
        If no data is specified, the name is used as data. 
        
        See the views.py module for an example.
        """
        QtGui.QAction.__init__(self,name,None)
        if icon:
            self.setIcon(icon)
        if signal is None:
            signal = DAction.signal
        self.signal = signal
        if data is None:
            data = name
        self.setData(QtCore.QVariant(data))
        self.connect(self,QtCore.SIGNAL("triggered()"),self.activated)
        
    def activated(self):
        self.emit(QtCore.SIGNAL(self.signal), str(self.data().toString()))


class ActionList(object):
    """Menu and toolbar with named actions.

    An action list is a list of strings, each connected to some action.
    The actions can be presented in a menu and/or a toolbar.
    On activating one of the menu or toolbar buttons, a given signal is
    emitted with the button string as parameter. A fixed function can be
    connected to this signal to act dependent on the string value.
    """

    def __init__(self,actions=[],function=None,menu=None,toolbar=None,icons=None):
        """Create an new action list, empty by default.

        A list of strings can be passed to initialize the actions.
        If a menu and/or toolbar are passed, a button is added to them
        for each string in the action list.
        If a function is passed, it will be called with the string as
        parameter when the item is triggered.

        If no icon names are specified, they are taken equal to the
        action names. Icons will be taken from the installed icon directory.
        If you want to specify other icons, use the add() method.
        """
        self.actions = []
        self.function = function
        self.menu = menu
        self.toolbar = toolbar
        if icons is None:
            icons = actions
        icons = map(utils.findIcon,icons)
        for name,icon in zip(actions,icons):
            self.add(name,icon)


    def add(self,name,icon=None):
        """Add a new name to the actions list and create a matching DAction.

        If the actions list has an associated menu or toolbar,
        a matching button will be inserted in each of these.
        If an icon is specified, it will be used on the menu and toolbar.
        The icon is either a filename or a QIcon object. 
        """
        if type(icon) == str:
            #print("CREATE ICON %s" % icon)
            if os.path.exists(icon):
                icon = QtGui.QIcon(QtGui.QPixmap(icon))
            else:
                raise RuntimeError,'Icons not installed properly'
        menutext = '&' + name.capitalize()
        a = DAction(menutext,icon,name)
        if self.function:
            QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),self.function)
        self.actions.append([name,a])
        if self.menu:
            self.menu.addAction(a)
        if self.toolbar:
            self.toolbar.addAction(a)


    def names(self):
        """Return an ordered list of names of the action items."""
        return [ i[0] for i in self.actions ]
#End
