#!/usr/bin/env python
# $Id$
"""A collection of custom widgets used in the pyFormex GUI"""

import qt,types


class MyQAction(qt.QAction):
    """A MyQAction is a QAction that sends a string as parameter when clicked."""
    def __init__(self,text,*args):
        qt.QAction.__init__(self,*args)
        self.signal = text
        self.connect(self,qt.SIGNAL("activated()"),self.activated)
        
    def activated(self):
        self.emit(qt.PYSIGNAL("Clicked"), (self.signal,))


class FileSelectionDialog(qt.QFileDialog):
    """A file selection dialog widget.

    You can specify a default path/filename that will be suggested initially.
    If a pattern is specified, only matching files will be shown.
    A pattern can be something like 'Images (*.png *.jpg)'.
    Default mode is to accept only existing files. You can specify
    any QFileDialog mode (e.g. QFileDialog.AnyFile to accept new files)
    
    """
    def __init__(self,default=None,pattern=None,mode=qt.QFileDialog.ExistingFile):
        qt.QFileDialog.__init__(self,default,pattern)
        self.setMode(mode)
        self.show()
        
    def getFilename(self):
        self.exec_loop()
        if self.result() == qt.QDialog.Accepted:
            return str(self.selectedFile())
        else:
            return None


class ConfigDialog(qt.QDialog):
    """A dialog widget to set the value of one or more items."""
    
    def __init__(self,cfgitems):
        """Creates a dialog which asks the user for the value cfgitems.

        cfgitems is the list of items that can be configured.
        Each item is a tuple holding at least the name of the item,
        and possible some more elements that limit the type of data
        that can be entered.
        name,value,type,range,default
        It should fit one of the following schemes:
        ('name') or ('name',str) : type string, any string input allowed
        ('name',int) : type int, any integer value allowed
        ('name',int,'min','max') : type int, only min <= value <= max allowed
        For each item a label with the name and a LineEdit widget are created,
        with a validator function where appropriate.
        """
        qt.QDialog.__init__(self,None,None,True)
        self.resize(400,200)
        self.setCaption("Config")
        self.fields = []
        self.result = []
        tab = qt.QVBoxLayout(self,11,6)
        for item in cfgitems:
            line = qt.QHBoxLayout(None,0,6)
            label = qt.QLabel(item[0],self)
            line.addWidget(label)
            input = qt.QLineEdit(str(item[1]),self)
            if len(item) == 2 or item[2] == 'str':
                print "%s is a string"%item[0]
            elif item[2] == 'int':
                print "%s is an integer"%item[0]
                if len(item) ==3 :
                    input.setValidator(qt.QIntValidator(input))
                else:
                    input.setValidator(qt.QIntValidator(item[3][0],item[3][1],input))
            elif item[2] == 'float':
                print "%s is a float"%item[0]
            input.selectAll()
            line.addWidget(input)
            self.fields.append([label,input])
            tab.addLayout(line)
        # add OK and Cancel buttons
        but = qt.QHBoxLayout(None,0,6)
        spacer = qt.QSpacerItem(0,0,qt.QSizePolicy.Expanding, qt.QSizePolicy.Minimum )
        but.addItem(spacer)
        ok = qt.QPushButton("OK",self)
        ok.setDefault(True)
        cancel = qt.QPushButton("CANCEL",self)
        cancel.setAccel(qt.QKeyEvent.Key_Escape)
        #cancel.setDefault(True)
        but.addWidget(cancel)
        but.addWidget(ok)
        tab.addLayout(but)
        self.connect(cancel,qt.SIGNAL("clicked()"),self,qt.SLOT("reject()"))
        self.connect(ok,qt.SIGNAL("clicked()"),self.acceptdata)
        
    def acceptdata(self):
        for label,input in self.fields:
            self.result.append([str(label.text()),str(input.text())])
        self.accept()
        
    def process(self):
        print "Now accepting events"
        if self.exec_loop() == qt.QDialog.Accepted:
            print "ACCEPT the following values"
            print self.result
        else:
            print "CHANGES REFUSED ",self.result
        return self.result
