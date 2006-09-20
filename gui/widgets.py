#!/usr/bin/env python
# $Id$
"""A collection of custom widgets used in the pyFormex GUI"""

import types
from PyQt4 import QtCore, QtGui


class FileSelection(QtGui.QFileDialog):
    """A file selection dialog widget.

    You can specify a default path/filename that will be suggested initially.
    If a pattern is specified, only matching files will be shown.
    A pattern can be something like 'Images (*.png *.jpg)'.
    Default mode is to accept any filename. You can specify exist=True
    to accept only existing files.
    
    """
    def __init__(self,dir,pattern=None,exist=False):
        """The constructor shows the widget."""
        QtGui.QFileDialog.__init__(self)
        dir = "."
        self.setDirectory(dir)
        if exist:
            mode = QtGui.QFileDialog.ExistingFile
            caption = "Open existing file"
        else:
            mode = QtGui.QFileDialog.AnyFile
            caption = "Save file as"
        self.setFileMode(mode)
        self.setWindowTitle(caption)
        self.show()
        
    def getFilename(self):
        """Ask for a filename by user interaction.

        Return the filename selected by the user.
        If the user hits CANCEL or ESC, None is returned.
        """
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            return str(self.selectedFiles()[0])
        else:
            return None


# !! The QtGui.QColorDialog can not be instantiated or subclassed.
# !! The color selection dialog is created by the static getColor
# !! function.

def getColor(col=None):
    """Create a color selection dialog and return the selected color.

    col is the initial selection.
    If a valid color is selected, its string name is returned, usually as
    a hex #RRGGBB string. If the dialog is canceled, None is returned.
    """
    col = QtGui.QColorDialog.getColor(QtGui.QColor(col))
    if col.isValid():
        return str(col.name())
    else:
        return None


## !! THIS IS NOT FULLY FUNCTIONAL YET
## It can already be used for string items  
class inputDialog(QtGui.QDialog):
    """A dialog widget to set the value of one or more items.

    This feature is still experimental (though already used in a few places.
    """
    
    def __init__(self,items,caption='Input Dialog',*args):
        """Creates a dialog which asks the user for the value of items.

        Each item in the 'items' list is a tuple holding at least the name
        of the item, and optionally some more elements that limit the type
        of data that can be entered. The general format of an item is:
          name,value,type,range,default
        It should fit one of the following schemes:
        ('name') or ('name',str) : type string, any string input allowed
        ('name',int) : type int, any integer value allowed
        ('name',int,'min','max') : type int, only min <= value <= max allowed
        For each item a label with the name and a LineEdit widget are created,
        with a validator function where appropriate.
        """
        QtGui.QDialog.__init__(self,*args)
        self.resize(400,200)
        self.setWindowTitle(caption)
        self.fields = []
        self.result = []
        form = QtGui.QVBoxLayout()
        for item in items:
            # Create the text label
            label = QtGui.QLabel(item[0])
            # Create the input field
            input = QtGui.QLineEdit(str(item[1]))
            if len(item) == 2 or item[2] == 'str':
                pass
                #print "%s is a string"%item[0]
            elif item[2] == 'int':
                #print "%s is an integer"%item[0]
                if len(item) ==3 :
                    input.setValidator(QtGui.QIntValidator(input))
                else:
                    input.setValidator(QtGui.QIntValidator(item[3][0],item[3][1],input))
            elif item[2] == 'float':
                pass
                #print "%s is a float"%item[0]
            input.selectAll()
            self.fields.append([label,input])
            # Add label and input field to a horizontal layout in the form
            line = QtGui.QHBoxLayout()
            line.addWidget(label)
            line.addWidget(input)
            form.addLayout(line)
        # add OK and Cancel buttons
        but = QtGui.QHBoxLayout()
        spacer = QtGui.QSpacerItem(0,0,QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum )
        but.addItem(spacer)
        ok = QtGui.QPushButton("OK",self)
        ok.setDefault(True)
        cancel = QtGui.QPushButton("CANCEL",self)
        #cancel.setAccel(QtGui.QKeyEvent.Key_Escape)
        #cancel.setDefault(True)
        but.addWidget(cancel)
        but.addWidget(ok)
        form.addLayout(but)
        self.connect(cancel,QtCore.SIGNAL("clicked()"),self,QtCore.SLOT("reject()"))
        self.connect(ok,QtCore.SIGNAL("clicked()"),self.acceptdata)
        self.setLayout(form)
        # Set the keyboard focus to the first input field
        self.setFocusProxy(self.fields[0][1])
        self.fields[0][0].setFocus()
        self.show()
        
    def acceptdata(self):
        for label,input in self.fields:
            self.result.append([str(label.text()),str(input.text())])
        self.accept()
        
    def process(self):
        accept = self.exec_() == QtGui.QDialog.Accepted
        return (self.result, accept)
