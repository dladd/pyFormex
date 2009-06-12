# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""A collection of custom widgets used in the pyFormex GUI"""

import os,types
from PyQt4 import QtCore, QtGui
import pyformex as GD
import colors
import odict
import imageViewer
import utils


# timeout value for all widgets providing timeout feature
#  (currently: InputDialog, MessageBox)

input_timeout = -1


class Options:
    pass

class FileSelection(QtGui.QFileDialog):
    """A file selection dialog widget.

    You can specify a default path/filename that will be suggested initially.
    If a pattern is specified, only matching files will be shown.
    A pattern can be something like 'Images (*.png *.jpg)' or a list
    of such strings.
    Default mode is to accept any filename. You can specify exist=True
    to accept only existing files. Or set exist=True and multi=True to
    accept multiple files.
    If dir==True, a single existing directory is asked.
    """
    
    def __init__(self,path,pattern=None,exist=False,multi=False,dir=False):
        """The constructor shows the widget."""
        QtGui.QFileDialog.__init__(self)
        #print path
        if os.path.isfile(path):
            #print "path is a file"
            self.setDirectory(os.path.dirname(path))
            self.selectFile(path)
        else:
            #print "path is a dir"
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
        
    def getFilename(self):
        """Ask for a filename by user interaction.

        Return the filename selected by the user.
        If the user hits CANCEL or ESC, None is returned.
        """
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
    """A file selection dialog specialized for opnening projects."""
    def __init__(self,path=None,pattern=None,exist=False,compression=0):
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
               
        self.leg = QtGui.QCheckBox("Allow Opening Legacy Format")
        self.leg.setToolTip("Check this box to allow opening projects saved in the headerless legacy format.")
        grid.addWidget(self.leg,nr,0,1,-1)


    def getResult(self):
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            opt = odict.ODict()
            opt.fn = str(self.selectedFiles()[0])
            opt.leg = self.leg.isChecked()
            if hasattr(self,'cpr'):
                opt.cpr = self.cpr.value()
            else:
                opt.cpr = 0
            return opt
        else:
            return None


class SaveImageDialog(FileSelection):
    """A file selection dialog with extra fields."""
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
    Returns a font if the user exited the dialog with the OK button.
    Returns None if the user clicked CANCEL.
    """
    font,ok = QtGui.QFontDialog.getFont()
    if ok:
        return font
    else:
        return None

        
class AppearenceDialog(QtGui.QDialog):
    """A dialog for setting the GUI appearance."""
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
        self.connect(acceptButton,QtCore.SIGNAL("clicked()"),self,QtCore.SLOT("accept()"))
        cancelButton = QtGui.QPushButton('Cancel')
        self.connect(cancelButton,QtCore.SIGNAL("clicked()"),self,QtCore.SLOT("reject()"))
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
    
    def __init__(self,slist=[],title='Selection Dialog',mode=None,sort=False,func=None):
        """Create the SelectionList dialog.
        """
        QtGui.QDialog.__init__(self)
        self.setWindowTitle(title)
        # Selection List
        self.listw = QtGui.QListWidget()
        self.listw.setMaximumWidth(100)
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
        self.connect(acceptButton,QtCore.SIGNAL("clicked()"),self,QtCore.SLOT("accept()"))
        cancelButton = QtGui.QPushButton('Cancel')
        self.connect(cancelButton,QtCore.SIGNAL("clicked()"),self,QtCore.SLOT("reject()"))
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
    col = QtGui.QColorDialog.getColor(col)
    if col.isValid():
        return str(col.name())
    else:
        return None

#####################################################################
########### General Input Dialog ####################################

class InputItem(QtGui.QHBoxLayout):
    """A single input item, usually with a label in front.

    The created widget is a QHBoxLayout which can be embedded in the vertical
    layout of a dialog.
    
    This is a super class, which just creates the label. The input
    field(s) should be added by a dedicated subclass.

    This class also defines default values for the name() and value()
    methods.

    Subclasses should override:
    - name(): if they called the superclass __init__() method without a name;
    - value(): if they did not create a self.input widget who's text() is
      the return value of the item.
    - setValue(): always, unless the field is readonly.

    Subclases can set validators on the input, like
      input.setValidator(QtGui.QIntValidator(input))
    Subclasses can define a show() method e.g. to select the data in the
    input field on display of the dialog.
    """
    
    def __init__(self,name=None,*args):
        """Creates a new inputitem with a name label in front.
        
        If a name is given, a label is created and added to the layout.
        """
        QtGui.QHBoxLayout.__init__(self,*args)
        if name:
            self.label = QtGui.QLabel(name)
            self.addWidget(self.label)

    def name(self):
        """Return the widget's name."""
        return str(self.label.text())

    def value(self):
        """Return the widget's value."""
        return str(self.input.text())

    def setValue(self,val):
        """Change the widget's value."""
        pass


class InputInfo(InputItem):
    """An unchangeable input item.
    """
    def __init__(self,name,value,*args):
        """Creates a new info field with a label in front.

        The info input field is an unchangeable text field.
        """
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QLineEdit(str(value))
        self.input.setReadOnly(True)
        self._value_ = value
        self.addWidget(self.input)

    def value(self):
        """Return the widget's value."""
        return self._value_

    def setValue(self,val):
        """Change the widget's value."""
        self.input.setText(str(val))


class InputString(InputItem):
    """A string input item."""
    
    def __init__(self,name,value,*args):
        """Creates a new string input field with a label in front.

        If the type of value is not a string, the input string
        will be eval'ed before returning.
        """
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QLineEdit(str(value))
        self._is_string_ = type(value) == str
        self.addWidget(self.input)

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
    """A scrollable text input item."""
    
    def __init__(self,name,value,*args):
        """Creates a new text input field with a label in front.

        If the type of value is not a string, the input text
        will be eval'ed before returning.
        """
        InputItem.__init__(self,name,*args)
        self._is_string_ = type(value) == str
        self.input =  QtGui.QTextEdit()
        self.setValue(value)
        self.addWidget(self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        s = str(self.input.toPlainText())
        if self._is_string_:
            print "VALUE",s
            return s
        else:
            return eval(s)

    def setValue(self,val):
        """Change the widget's value."""
        self.input.setPlainText(str(val))


class InputBool(InputItem):
    """A boolean input item."""
    
    def __init__(self,name,value,*args):
        """Creates a new checkbox for the input of a boolean value.
        
        Displays the name next to a checkbox, which will initially be set
        if value evaluates to True. (Does not use the label)
        The value is either True or False,depending on the setting
        of the checkbox.
        """
        InputItem.__init__(self,None,*args)
        self.input = QtGui.QCheckBox(name)
        self.setValue(value)
        self.addWidget(self.input)

    def name(self):
        """Return the widget's name."""
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
    
    def __init__(self,name,choices,default,*args):
        """Creates a new combobox for the selection of a value from a list.

        choices is a list/tuple of possible values.
        default is the initial/default choice.
        If default is not in the choices list, it is prepended.
        If default is None, the first item of choices is taken as the default.
       
        The choices are presented to the user as a combobox, which will
        initially be set to the default value.
        """
        if default is None:
            default = choices[0]
        elif default not in choices:
            choices[0:0] = [ default ]
        self._choices_ = [ str(s) for s in choices ]
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QComboBox()
        self.input.addItems(self._choices_)
        self.setValue(default)
        self.addWidget(self.input)

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
    
    def __init__(self,name,choices,default,direction='h',*args):
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
        InputItem.__init__(self,name,*args)
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
        self.addWidget(self.input)

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
    
    def __init__(self,name,choices,default=None,direction='h',*args):
        """Creates pushbuttons for the selection of a value from a list.

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
        InputItem.__init__(self,name,*args)
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
        self.addWidget(self.input)

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
    'min', 'max': range of the scale (integer)
    """
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new integer input field with a label in front."""
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QLineEdit(str(value))
        self.validator = QtGui.QIntValidator(self)
        if kargs.has_key('min'):
            line.validator.setBottom(int(kargs['min']))
        if kargs.has_key('max'):
            line.validator.setTop(int(kargs['max']))
        self.input.setValidator(self.validator)
        self.addWidget(self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
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
    
    def __init__(self,name,value,*args):
        """Creates a new float input field with a label in front."""
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QLineEdit(str(value))
        self.validator = QtGui.QDoubleValidator(self)
        self.input.setValidator(self.validator)
        self.addWidget(self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
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
      'min', 'max': range of the scale (integer)
      'ticks'     : step for the tick marks (default range length / 10)
      'func'      : an optional function to be called whenever the value is
                    changed. The function takes a float/integer argument.
    """
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new integer input slider."""
        InputInteger.__init__(self,name,value,*args)
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
      'min', 'max': range of the scale (integer)
      'ticks'     : step for the tick marks (default range length / 10)
      'func'      : an optional function to be called whenever the value is
                    changed. The function takes a float/integer argument.
    """
    
    def __init__(self,name,value,*args,**kargs):
        """Creates a new integer input slider."""
        InputFloat.__init__(self,name,value,*args)
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
    
    def __init__(self,name,value,*args):
        """Creates a new color input field with a label in front.

        The color input field is a button displaying the current color.
        Clicking on the button opens a color dialog, and the returned
        value is set in the button.
        """
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QPushButton(colors.colorName(value))
        self.connect(self.input,QtCore.SIGNAL("clicked()"),self.setColor)
        self.addWidget(self.input)

    def setColor(self):
        color = getColor(self.input.text())
        if color:
            self.input.setText(str(color))

    def setValue(self,value):
        """Change the widget's value."""
        pass



class InputDialog(QtGui.QDialog):
    """A dialog widget to set the value of one or more items.

    While general input dialogs can be constructed from all the underlying
    Qt classes, this widget provides a way to construct fairly complex
    input dialogs with a minimum of effort.
    """
    
    def __init__(self,items,caption=None,parent=None,flags=None,actions=None,default=None,report_pos=False):
        """Creates a dialog which asks the user for the value of items.

        Each item in the 'items' list is a tuple holding at least the name
        of the item, and optionally some more elements that limit the type
        of data that can be entered. The general format of an item is:
          name,value,type,range
        It should fit one of the following schemes:
        ('name',str) : type string, any string input allowed
        ('name',int) : type int, any integer value allowed
        ('name',int,'min','max') : type int, only min <= value <= max allowed
        For each item a label with the name and a LineEdit widget are created,
        with a validator function where appropriate.

        Input items are defined by a list with the following structure:
        [ name, value, type, range... ]
        The fields have the following meaning:
          name:  the name of the field,
          value: the initial or default value of the field,
          type:  the type of values the field can accept,
          range: the range of values the field can accept,
        The first two fields are mandatory. In many cases the type can be
        determined from the value and no other fields are required. Thus:
        [ 'name', 'value' ] will accept any string (initial string = 'value'),
        [ 'name', True ] will show a checkbox with the item checked,
        [ 'name', 10 ] will accept any integer,
        [ 'name', 1.5 ] will accept any float.

        Range settings for int and float types:
        [ 'name', 1, int, 0, 4 ] will accept an integer from 0 to 4, inclusive;
        [ 'name', 1, float, 0.0, 1.0, 2 ] will accept a float in the range
           from 0.0 to 1.0 with a maximum of two decimals.

        Composed types:
        [ 'name', 'option1', 'select', ['option0','option1','option2']] will
        present a combobox to select between one of the options.
        The initial and default value is 'option1'.

        [ 'name', 'option1', 'radio', ['option0','option1','option2']] will
        present a group of radiobuttons to select between one of the options.
        The initial and default value is 'option1'.
        A variant 'vradio' aligns the options vertically. 
        
        [ 'name', 'option1', 'push', ['option0','option1','option2']] will
        present a group of pushbuttons to select between one of the options.
        The initial and default value is 'option1'.
        A variant 'vpush' aligns the options vertically. 

        [ 'name', 'red', 'color' ] will present a color selection widget,
        with 'red' as the initial choice.
        """
        if parent is None:
            parent = GD.GUI
        QtGui.QDialog.__init__(self,parent)
        if caption is None:
            caption = 'pyFormex-dialog'
        self.setWindowTitle(str(caption))
        self.fields = []
        self.result = {}
        self.timedOut = False
        self.report_pos = report_pos
        form = QtGui.QVBoxLayout()
        for item in items:
            name,value = item[:2]
            if len(item) > 2:
                itemtype = item[2]
            else:
                itemtype = type(value)
            #print itemtype
            options = {}
            if len(item) > 3 and type(item[3] == dict):
                options = item[3]
                
            if itemtype == bool:
                line = InputBool(name,value)

            elif itemtype == int:
                line = InputInteger(name,value)
                if len(item) > 3 and type(item[3] != dict):
                    options['min'] = int(item[3])
##                     line.validator.setBottom(int(item[3]))
                if len(item) > 4:
                    options['max'] = int(item[4])
##                     line.validator.setTop(int(item[4]))

            elif itemtype == float:
                line = InputFloat(name,value)
                if len(item) > 3:
                    line.validator.setBottom(float(item[3]))
                if len(item) > 4:
                    line.validator.setTop(float(item[4]))
                if len(item) > 5:
                    line.validator.setDecimals(int(item[5]))

            elif itemtype == 'slider':
                if type(value) == int:
                    line = InputSlider(name,value,**options)
                elif type(value) == float:
                    line = InputFSlider(name,value,**options)
 
            elif itemtype == 'info':
                line = InputInfo(name,value)

            elif itemtype == 'text':
                line = InputText(name,value)

            elif itemtype == 'color':
                line = InputColor(name,value)

            elif itemtype == 'select' :
                if len(item) > 3:
                    choices = item[3]
                else:
                    choices = []
                line = InputCombo(name,choices,value)

            elif itemtype in ['radio','hradio','vradio']:
                if len(item) > 3:
                    choices = item[3]
                else:
                    choices = []
                line = InputRadio(name,choices,value,direction=itemtype[0])

            elif itemtype in ['push','hpush','vpush']:
                if len(item) > 3:
                    choices = item[3]
                else:
                    choices = []
                line = InputPush(name,choices,value,direction=itemtype[0])

            else: # Anything else is handled as a string
                #itemtype = str:
                line = InputString(name,value)
                
            form.addLayout(line)
            self.fields.append(line)

        # add the action buttons
        if actions is None:
            actions = [('CANCEL',),('OK',)]
            default = 'OK'
        but = dialogButtons(self,actions,default)
        self.connect(self,QtCore.SIGNAL("accepted()"),self.acceptData)
        form.addLayout(but)
        self.setLayout(form)
        # Set the keyboard focus to the first input field
        self.fields[0].input.setFocus()
        self.show()


    def __getitem__(self,name):
        """Return the input item with specified name."""
        items = [ f for f in self.fields if f.name() == name ]
        if len(items) > 0:
            return items[0]
        else:
            return None
        
        
    def acceptData(self):
        """Update the dialog's return value from the field values.

        This function is connected to the 'accepted()' signal.
        Modal dialogs should normally not need to call it.
        In non-modal dialogs however, you can call it to update the
        results without having to raise the accepted() signal (which
        would close the dialog).
        """
        self.result = {}
        self.result.update([ (fld.name(),fld.value()) for fld in self.fields ])
        print "RESULT",self.result
        if self.report_pos:
            self.result.update({'__pos__':self.pos()})
        self.accepted = True
        

    def updateData(self,d):
        """Update a dialog from the data in given dictionary.

        d is a dictionary where the keys are field names in t the dialog.
        The values will be set in the corresponding input items.
        """
        for f in self.fields:
            n = f.name()
            if n in d:
                f.setValue(d[n])
            

    def timeout(self):
        self.timedOut = True
        
        
    def getResult(self,timeout=None,timeoutAccept=True):
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

        This function also sets the exit mode, so that the caller can test how
        the dialog was ended.
        self.accepted == TRUE/FALSE
        self.timedOut == TRUE/FALSE
        """
        self.accepted = False
        GD.debug("TIMEOUT %s" % timeout)
        if timeout is None:
            timeout = input_timeout

        GD.debug("TIMEOUT %s" % timeout)
        if timeout >= 0:
            # Start the timer:
            GD.debug("START TIMEOUT %s" % timeout)
            try:
                timeout = float(timeout)
                if timeout >= 0.0:
                    timer = QtCore.QTimer()
                    if timeoutAccept == True:
                        timeoutFunc = self.accept
                    else:
                        timeoutFunc = self.reject
                    timer.connect(timer,QtCore.SIGNAL("timeout()"),self.timeout)
                    timer.connect(timer,QtCore.SIGNAL("timeout()"),timeoutFunc)
                    timer.setSingleShot(True)
                    timeout = int(1000*timeout)
                    timer.start(timeout)
            except:
                raise
            
        self.exec_()
        self.activateWindow()
        self.raise_()
        GD.app.processEvents()
        return self.result




class NewInputDialog(QtGui.QDialog):
    """A dialog widget to set the value of one or more items.

    While general input dialogs can be constructed from all the underlying
    Qt classes, this widget provides a way to construct fairly complex
    input dialogs with a minimum of effort.
    """
    
    def __init__(self,items,caption=None,parent=None,flags=None,actions=None,default=None,report_pos=False):
        """Creates a dialog which asks the user for the value of items.

        Each item in the 'items' list is a tuple holding at least the name
        of the item, and optionally some more elements that limit the type
        of data that can be entered. The general format of an item is:
          name,value,type,range
        It should fit one of the following schemes:
        ('name',str) : type string, any string input allowed
        ('name',int) : type int, any integer value allowed
        ('name',int,'min','max') : type int, only min <= value <= max allowed
        For each item a label with the name and a LineEdit widget are created,
        with a validator function where appropriate.

        Input items are defined by a list with the following structure:
        [ name, value, type, range... ]
        The fields have the following meaning:
          name:  the name of the field,
          value: the initial or default value of the field,
          type:  the type of values the field can accept,
          range: the range of values the field can accept,
        The first two fields are mandatory. In many cases the type can be
        determined from the value and no other fields are required. Thus:
        [ 'name', 'value' ] will accept any string (initial string = 'value'),
        [ 'name', True ] will show a checkbox with the item checked,
        [ 'name', 10 ] will accept any integer,
        [ 'name', 1.5 ] will accept any float.

        Range settings for int and float types:
        [ 'name', 1, int, 0, 4 ] will accept an integer from 0 to 4, inclusive;
        [ 'name', 1, float, 0.0, 1.0, 2 ] will accept a float in the range
           from 0.0 to 1.0 with a maximum of two decimals.

        Composed types:
        [ 'name', 'option1', 'select', ['option0','option1','option2']] will
        present a combobox to select between one of the options.
        The initial and default value is 'option1'.

        [ 'name', 'option1', 'radio', ['option0','option1','option2']] will
        present a group of radiobuttons to select between one of the options.
        The initial and default value is 'option1'.
        A variant 'vradio' aligns the options vertically. 
        
        [ 'name', 'option1', 'push', ['option0','option1','option2']] will
        present a group of pushbuttons to select between one of the options.
        The initial and default value is 'option1'.
        A variant 'vpush' aligns the options vertically. 

        [ 'name', 'red', 'color' ] will present a color selection widget,
        with 'red' as the initial choice.
        """
        if parent is None:
            parent = GD.GUI
        QtGui.QDialog.__init__(self,parent)
        if caption is None:
            caption = 'pyFormex-dialog'
        self.setWindowTitle(str(caption))
        self.fields = []
        self.result = {}
        self.report_pos = report_pos
        form = QtGui.QVBoxLayout()
        for item in items:
            name,value = item[:2]
            if len(item) > 2:
                itemtype = item[2]
            else:
                itemtype = type(value)
            #print itemtype
            options = {}
            if len(item) > 3 and type(item[3] == dict):
                options = item[3]
                
            if itemtype == bool:
                line = InputBool(name,value)

            elif itemtype == int:
                line = InputInteger(name,value)
                if len(item) > 3 and type(item[3] != dict):
                    options['min'] = int(item[3])
##                     line.validator.setBottom(int(item[3]))
                if len(item) > 4:
                    options['max'] = int(item[4])
##                     line.validator.setTop(int(item[4]))

            elif itemtype == float:
                line = InputFloat(name,value)
                if len(item) > 3:
                    line.validator.setBottom(float(item[3]))
                if len(item) > 4:
                    line.validator.setTop(float(item[4]))
                if len(item) > 5:
                    line.validator.setDecimals(int(item[5]))

            elif itemtype == 'slider':
                if type(value) == int:
                    line = InputSlider(name,value,**options)
                elif type(value) == float:
                    line = InputFSlider(name,value,**options)
 
            elif itemtype == 'info':
                line = InputInfo(name,value)

            elif itemtype == 'color':
                line = InputColor(name,value)

            elif itemtype == 'select' :
                if len(item) > 3:
                    choices = item[3]
                else:
                    choices = []
                line = InputCombo(name,choices,value)

            elif itemtype in ['radio','hradio','vradio']:
                if len(item) > 3:
                    choices = item[3]
                else:
                    choices = []
                line = InputRadio(name,choices,value,direction=itemtype[0])

            elif itemtype in ['push','hpush','vpush']:
                if len(item) > 3:
                    choices = item[3]
                else:
                    choices = []
                line = InputPush(name,choices,value,direction=itemtype[0])

            else: # Anything else is handled as a string
                #itemtype = str:
                line = InputString(name,value)
                
            form.addLayout(line)
            self.fields.append(line)

        # add the action buttons
        if actions is None:
            actions = [('CANCEL',),('OK',)]
            default = 'OK'
        but = dialogButtons(self,actions,default)
        self.connect(self,QtCore.SIGNAL("accepted()"),self.acceptData)
        form.addLayout(but)
        self.setLayout(form)
        # Set the keyboard focus to the first input field
        self.fields[0].input.setFocus()
        self.show()


    def __getitem__(self,name):
        """Return the input item with specified name."""
        items = [ f for f in self.fields if f.name() == name ]
        if len(items) > 0:
            return items[0]
        else:
            return None
        
        
    def acceptData(self):
        """Update the dialog's return value from the field values.

        This function is connected to the 'accepted()' signal.
        Modal dialogs should normally not need to call it.
        In non-modal dialogs however, you can call it to update the
        results without having to raise the accepted() signal (which
        would close the dialog).
        """
        self.result = {}
        self.result.update([ (fld.name(),fld.value()) for fld in self.fields ])
        if self.report_pos:
            self.result.update({'__pos__':self.pos()})
        self.accepted = True
        

    def updateData(self,d):
        """Update a dialog from the data in given dictionary.

        d is a dictionary where the keys are field names in t the dialog.
        The values will be set in the corresponding input items.
        """
        for f in self.fields:
            n = f.name()
            if n in d:
                f.setValue(d[n])
            

    def timeout(self):
        self.timedOut = True
        
        
    def getResult(self,timeout=None,timeoutAccept=True):
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

        This function also sets the exit mode, so that the caller can test how
        the dialog was ended.
        self.accepted == TRUE/FALSE
        self.timedOut == TRUE/FALSE
        """
        self.timedOut = False
        self.accepted = False
        if timeout is None:
            timeout = input_timeout

        if timeout >= 0:
            # Start the timer:
            try:
                timeout = float(timeout)
                if timeout >= 0.0:
                    timer = QtCore.QTimer()
                    if timeoutAccept == True:
                        timeoutFunc = self.accept
                    else:
                        timeoutFunc = self.reject
                    timer.connect(timer,QtCore.SIGNAL("timeout()"),self.timeout)
                    timer.connect(timer,QtCore.SIGNAL("timeout()"),timeoutFunc)
                    timer.setSingleShot(True)
                    timeout = int(1000*timeout)
                    timer.start(timeout)
            except:
                raise
            
        self.exec_()
        self.activateWindow()
        self.raise_()
        GD.app.processEvents()
        return self.result


def dialogButtons(dialog,actions,default):
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
            slot = (dialog,QtCore.SLOT("accept()"))
        elif n == 'cancel':
            slot = (dialog,QtCore.SLOT("reject()"))
        else:
            slot = (dialog,QtCore.SLOT("reject()"))
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
            caption = 'pyFormex-dialog'
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
# Some static functions for displaying text widgets

def messageBox(message,level='info',choices=['OK'],default=None,timeout=None):
    """Display a message box and wait for user response.

    The message box displays a text, an icon depending on the level
    (either 'about', 'info', 'warning' or 'error') and 1-3 buttons
    with the specified action text. The 'about' level has no buttons.

    The function returns the text of the button that was clicked or
    an empty string is ESC was hit.
    """
    if default is None:
        default = choices[-1]
    w = QtGui.QMessageBox()
    w.setText(message)
    if level == 'error':
        w.setIcon(QtGui.QMessageBox.Critical)
    elif level == 'warning':
        w.setIcon(QtGui.QMessageBox.Warning)
    elif level == 'info':
        w.setIcon(QtGui.QMessageBox.Information)
    elif level == 'question':
        w.setIcon(QtGui.QMessageBox.Question)
    for a in choices:
        b = w.addButton(a,QtGui.QMessageBox.AcceptRole)
        if a == default:
            w.setDefaultButton(b)
            
    if timeout is None:
        timeout = input_timeout

    # Start the timer:
    if timeout >= 0:
        GD.debug("STARTING TIIMEOUT TIMER %s" % input_timeout)
        try:
            timeout = float(timeout)
            if timeout >= 0.0:
                timer = QtCore.QTimer()
                timer.connect(timer,QtCore.SIGNAL("timeout()"),w,QtCore.SLOT("accept()"))
                timer.setSingleShot(True)
                timeout = int(1000*timeout)
                timer.start(timeout)
        except:
            raise
            
    w.exec_()
    b = w.clickedButton()
    if b == 0:
        b = w.defaultButton()
    if b:
        return str(b.text())
    else:
        return ''


def textBox(text,type=None,choices=['OK']):
    """Display a text and wait for user response.

    Possible choices are 'OK' and 'CANCEL'.
    The function returns True if the OK button was clicked or 'ENTER'
    was pressed, False if the 'CANCEL' button was pressed or ESC was pressed.
    """
    w = QtGui.QDialog()
    w.setWindowTitle('pyFormex Text Display')
    t = QtGui.QTextEdit()
    t.setReadOnly(True)
    if type == 'plain':
        t.setPlainText(text)
    elif type == 'html':
        w.setWindowTitle('pyFormex Html Display')
        t.setHtml(text)
    else:
        t.setText(text)
    bl = QtGui.QHBoxLayout()
    bl.addStretch()
    if 'OK' in choices:
        b = QtGui.QPushButton('OK')
        QtCore.QObject.connect(b,QtCore.SIGNAL("clicked()"),w,QtCore.SLOT("accept()"))
        bl.addWidget(b)
    if 'CANCEL' in choices:
        b = QtGui.QPushButton('CANCEL')
        QtCore.QObject.connect(b,QtCore.SIGNAL("clicked()"),w,QtCore.SLOT("reject()"))
        bl.addWidget(b)
    bl.addStretch()
    h = QtGui.QWidget()
    h.setLayout(bl)
    l = QtGui.QVBoxLayout()
    l.addWidget(t)
    l.addWidget(h)
    w.setLayout(l)
    w.resize(800,400)
    return w.exec_() == QtGui.QDialog.Accepted


############################# Named button box ###########################

class ButtonBox(QtGui.QWidget):
    def __init__(self,name,choices,funcs,*args):
        QtGui.QWidget.__init__(self,*args)
        s = InputPush(name,choices)
        s.setSpacing(0)
        s.setMargin(0)
        for r,f in zip(s.rb,funcs):
            self.connect(r,QtCore.SIGNAL("clicked()"),f)
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


############################# Named combo box ###########################

class ComboBox(QtGui.QWidget):
    def __init__(self,name,choices,func,*args):
        QtGui.QWidget.__init__(self,*args)
        s = InputCombo(name,choices,None,*args)
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

    This class is not intended for direct use, but through subclasses.
    Subclasses should implement at least the following methods:
      addSeparator()              insertSeperator(before)
      addAction(text,action)      insertAction(before,text,action)
      addMenu(text,menu)          insertMenu(before,text,menu)
      
    QtGui.Menu and QtGui.MenuBar provide these methods.
    """

    def __init__(self,title='AMenu',parent=None,before=None,items=None):
        """Create a menu.

        This is a hierarchical menu that keeps a list of its item
        names and actions.
        """
        self.title = title
        self.parent = parent
        if parent and isinstance(parent,BaseMenu):
            if before:
                before = parent.itemAction(before)
            parent.insert_menu(self,before)
            parent.menuitems.append((normalize(title),self))
        self.menuitems = []
        if items:
            self.insertItems(items)


    def item(self,text):
        """Get the menu item with given normalized text.

        Text normalization removes all '&' characters and
        converts to lower case.
        """
        return dict(self.menuitems).get(normalize(text),None)


    def itemAction(self,item):
        """Return the action corresponding to item.

        item is either one of the menu's item texts, or one of its
        values. This method guarantees that the return value is either the
        corresponding Action, or None.
        """
        if item not in dict(self.menuitems).values():
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
                #self.insert_menu(a,before)
            else:
                if type(val) == str:
                    val = eval(val)
                if len(item) > 2 and item[2].has_key('data'):
                    print "INSERT A DACTION", item
                    a = DAction(txt,data=item[2]['data'])
                    QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),val)
                    self.insert_action(a,before)
                else:
                    a = self.create_insert_action(txt,val,before)
                if len(item) > 2:
                    #print 'item = %s' % str(item)
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
            self.menuitems.append((normalize(txt),a))


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
            for i,item in enumerate(self.parent.menuitems):
                if item[1] == self:
                    del self.parent.menuitems[i]


class MenuBar(BaseMenu,QtGui.QMenuBar):
    """A menu bar allowing easy menu creation."""

    def __init__(self):
        """Create the menubar."""
        QtGui.QMenuBar.__init__(self)
        BaseMenu.__init__(self)


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
            #print "CREATE ICON %s" % icon
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
