# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""A collection of custom widgets used in the pyFormex GUI"""

import os,types
from PyQt4 import QtCore, QtGui
import globaldata as GD
import colors
import utils


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
        self.mul = QtGui.QCheckBox("Multi mode")
        self.hot = QtGui.QCheckBox("Activate 'S' hotkey")
        self.aut = QtGui.QCheckBox('Autosave mode')
        self.mul.setChecked(multi)
        self.hot.setChecked(multi)
        self.win.setToolTip("If checked, the whole window is saved;\nelse, only the Canvas is saved.")
        self.mul.setToolTip("If checked, multiple images can be saved\nwith autogenerated names.")
        self.hot.setToolTip("If checked, a new image can be saved\nby hitting the 'S' key when focus is in the Canvas.")
        self.aut.setToolTip("If checked, a new image will saved\non each draw() operation")
        grid.addWidget(self.win,nr,0)
        grid.addWidget(self.mul,nr,1)
        grid.addWidget(self.hot,nr,2)
        grid.addWidget(self.aut,nr,3)

    def getResult(self):
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            fn = str(self.selectedFiles()[0])
            wi = self.win.isChecked()
            mu = self.mul.isChecked()
            hk = self.hot.isChecked()
            as = self.aut.isChecked()
            return fn,wi,mu,hk,as
        else:
            return None,False,False,False,False


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

        
class Selection(QtGui.QDialog):
    """A dialog for selecting one or more items from a list."""
    
    selection_mode = {
        None: QtGui.QAbstractItemView.NoSelection,
        'single': QtGui.QAbstractItemView.SingleSelection,
        'multi': QtGui.QAbstractItemView.MultiSelection,
        'contiguous': QtGui.QAbstractItemView.ContiguousSelection,
        'extended': QtGui.QAbstractItemView.ExtendedSelection,
        }
    
    def __init__(self,slist=[],title='Selection Dialog',mode=None,sort=False,\
                 selected=[]):
        """Create the SelectionList dialog.

        selected is a list of items that are initially selected.
        """
        QtGui.QDialog.__init__(self)
        self.setWindowTitle(title)
        # Selection List
        self.listw = QtGui.QListWidget()
        self.listw.addItems(slist)
        if sort:
            self.listw.sortItems()
        if selected:
            self.setSelected(selected)
        self.listw.setSelectionMode(self.selection_mode[mode])
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
                self.listw.setItemSelected(i,True)
                # SHOULD BECOME:
                # i.setSelected(True) # requires Qt 4.2
                # i.setCheckState(QtCore.Qt.Checked)

                
    def getResult(self):
        self.exec_()
        if self.result() == QtGui.QDialog.Accepted:
            res = [ i.text() for i in self.listw.selectedItems() ]
            return map(str,res)
        else:
            return []
        

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

    The widget is a QHBoxLayout which can be embedded in the vertical
    layout of a dialog.
    
    This is a super class, which just creates the label. The input
    field(s) should be added by a dedicated subclass.

    This class also defines default values for the name() and value()
    methods.

    Subclasses should override:
    - name(): if they called the superclass __init__() method without a name;
    - value(): if they did not create a self.input widget who's text() is
      the return value of the item.

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


class InputString(InputItem):
    """A string input item."""
    
    def __init__(self,name,value,*args):
        """Creates a new string input field with a label in front.

        If the type of value is not a string, the input string
        will be eval'ed before returning.
        """
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QLineEdit(str(value))
        self.addWidget(self.input)
        self.str = type(value) == str

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        s = str(self.input.text())
        if self.str:
            return s
        else:
            return eval(s)


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
        if value:
            self.input.setCheckState(QtCore.Qt.Checked)
        else:
            self.input.setCheckState(QtCore.Qt.Unchecked)
        self.addWidget(self.input)

    def name(self):
        """Return the widget's name."""
        return str(self.input.text())

    def value(self):
        """Return the widget's value."""
        return self.input.checkState() == QtCore.Qt.Checked

    
class InputSelect(InputItem):
    """A selection InputItem."""
    
    def __init__(self,name,value,*args):
        """Creates a new combobox for the selection of a value from a list.

        value is a list/tuple of possible values.
        Displays the name next to a combobox, which will initially be set
        to the first value in the list.
        The value is always one either True or False,depending on the setting
        of the checkbox.
        """
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QComboBox()
        for v in value:
            self.input.addItem(str(v))
        self.addWidget(self.input)

    def value(self):
        """Return the widget's value."""
        return str(self.input.currentText())


class InputInteger(InputItem):
    """An integer input item."""
    
    def __init__(self,name,value,*args):
        """Creates a new integer input field with a label in front."""
        InputItem.__init__(self,name,*args)
        self.input = QtGui.QLineEdit(str(value))
        self.validator = QtGui.QIntValidator(self)
        self.input.setValidator(self.validator)
        self.addWidget(self.input)

    def show(self):
        """Select all text on first display.""" 
        InputItem.show(self,*args)
        self.input.selectAll()

    def value(self):
        """Return the widget's value."""
        return int(self.input.text())


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



class InputDialog(QtGui.QDialog):
    """A dialog widget to set the value of one or more items.

    This feature is still experimental (though already used in a few places.
    """
    
    def __init__(self,items,caption=None,*args):
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
        if caption is None:
            caption = 'pyFormex-input'
        self.setWindowTitle(str(caption))
        self.fields = []
        self.result = {}
        form = QtGui.QVBoxLayout()
        for item in items:
            name,value = item[:2]
            if len(item) > 2:
                itemtype = item[2]
            else:
                itemtype = type(value)
            GD.debug("INPUT ITEM %s TYPE %s" % (name,itemtype))
            if itemtype == bool:
                line = InputBool(name,value)

            elif itemtype == int:
                line = InputInteger(name,value)
                if len(item) > 3:
                    line.validator.setBottom(int(item[3]))
                if len(item) > 4:
                    line.validator.setTop(int(item[4]))

            elif itemtype == float:
                line = InputFloat(name,value)
                if len(item) > 3:
                    line.validator.setBottom(float(item[3]))
                if len(item) > 4:
                    line.validator.setTop(float(item[4]))
                if len(item) > 5:
                    line.validator.setDecimals(int(item[5]))

            elif itemtype == 'color':
                line = InputColor(name,value)

            elif itemtype == 'select' :
                line = InputSelect(name,value)
                if len(item) > 3:
                    line.input.setCurrentIndex(item[1].index(item[3]))

            else: # Anything else is handled as a string
                #itemtype = str:
                line = InputString(name,value)
                
            form.addLayout(line)
            self.fields.append(line)

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
        self.fields[0].input.setFocus()
        self.show()
        
    def acceptdata(self):
        self.result = {}
        self.result.update([ (fld.name(),fld.value()) for fld in self.fields ])
        self.accept()
        
    def getResult(self):
        accept = self.exec_() == QtGui.QDialog.Accepted
        GD.app.processEvents()
        return (self.result, accept)


############################# Menu ##############################


def addMenuItems(menu, items=[]):
    """Add a list of items to a menu.

    Each item is a tuple of two to five elements:
       Text, Action, [ Icon,  ShortCut, ToolTip ].

    Item text is the text that will be displayed in the menu. An optional '&'
    may be used to flag the next character as the shortcut key. The '&' will
    be stripped off before displaying the text. If the text starts with '---',
    a separator is created. 

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
    """
    for item in items:
        txt,val = item[:2]
        if txt.startswith('---') or val is None:
            menu.addSeparator()
        elif isinstance(val, list):
            pop = QtGui.QMenu(txt,menu)
            addMenuItems(pop,val)
            menu.addMenu(pop)
        else:
            if type(val) == str:
                val = eval(val)
            a = menu.addAction(txt,val)
            if len(item) > 2 and item[2]:
                a.setIcon(QtGui.QIcon(QtGui.QPixmap(utils.findIcon(item[2]))))
            if len(item) > 3 and item[3]:
                a.setShortcut(item[3])
            if len(item) > 4 and item[4]:
                a.setToolTip(item[4])


def normalize(s):
    """Normalize a string.

    Text normalization removes all '&' characters and converts to lower case.
    """
    return s.replace('&','').lower()

                 
def menuDict(menu):
    """Returns the menudict of a menu.

    The menudict holds the normalized text labels and corresponding actions.
    Text normalization removes all '&' characters and converts to lower case.
    """
    return dict([[normalize(str(a.text())),a] for a in menu.actions()])
    

def menuItem(menu, text):
    """Get the menu item with given normalized text.

    Text normalization removes all '&' characters and converts to lower case.
    """
    return menuDict(menu).get(normalize(text),None)


class Menu(QtGui.QMenu):
    """A popup menu for user actions."""

    def __init__(self,title='UserMenu',parent=None):
        """Create a popup/pulldown menu.

        If parent==None, the menu is a standalone popup menu.
        If parent is given, the menu will be inserted in the parent menu.
        If parent==GD.gui, the menu is inserted in the main menu bar.
        
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
        self.parent = parent
        if self.parent == GD.gui:
            GD.gui.menu.insertMenu(self)
        elif parent is None:
            self.setWindowFlags(QtCore.Qt.Dialog)
            self.setWindowTitle(title)
        self.done = False


    def addItems(self,itemlist):
        addMenuItems(self,itemlist)


    def process(self):
        if not self.done:
            if not self.insert:
                self.show()
            GD.app.processEvents()


    def close(self):
        """Close the menu."""
        self.done=True
        if self.parent == GD.gui:
            GD.gui.menu.removeMenu(str(self.title()))


class MenuBar(QtGui.QMenuBar):
    """A menu bar allowing easy menu creation."""

    def __init__(self):
        """Create the menubar."""
        QtGui.QMenuBar.__init__(self)


    def addItems(self,itemlist):
        addMenuItems(self,itemlist)


    def insertMenu(self,menu,before='help'):
        """Insert a menu in the menubar before the specified menu.

        The new menu can be inserted BEFORE any of the existing menus.
        By default the new menu will be inserted before the Help menu.
        """
        item = menuItem(self,before)
        if item:
            QtGui.QMenuBar.insertMenu(self,item,menu)
        else:
            GD.debug("No such menu item: %s" % before)


    def removeMenu(self,menu):
        """Remove a menu from the main menubar.

        menu is either a menu title or a menu action.
        """
        if type(menu) == str:
            menu = menuItem(self,menu)
        if menu:
            self.removeAction(menu)



###################### Action List ############################################

class DAction(QtGui.QAction):
    """A DAction is a QAction that emits a signal with a string parameter.

    When triggered, this action sends a signal (default 'Clicked') with a
    custom string as parameter. The connected slot can then act depending
    on this parameter.
    """

    signal = "Clicked"
    
    def __init__(self,name,icon=None,data=None):
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
        if not data:
            data = name
        self.setData(QtCore.QVariant(data))
        self.connect(self,QtCore.SIGNAL("triggered()"),self.activated)
        
    def activated(self):
        self.emit(QtCore.SIGNAL(DAction.signal), str(self.data().toString()))


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
        if type(icon) == str and os.path.exists(icon):
            icon = QtGui.QIcon(QtGui.QPixmap(icon))
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
