# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

import pyformex as pf
from gui import menu

from plugins.postproc import *
from plugins.fe_post import FeResult
from plugins.objects import Objects
from gui.colorscale import ColorScale,ColorLegend
import utils
from odict import ODict

import commands

from numpy import *
from formex import *
from gui.draw import *
from gui.colors import *

from PyQt4 import QtCore


class AttributeModel(QtCore.QAbstractTableModel):
    """A model representing the attributes of an object.

    """
    header = [ 'attribute', 'value', 'is a dict', 'has __dict__', '__class__' ]
    def __init__(self,name,dic=None,parent=None,*args): 
        QtCore.QAbstractItemModel.__init__(self,parent,*args) 
        if dic is None:
            dic = gobals()
        self.dic = dic
        self.name = name
        self.obj = dic.get(name,None)
        keys = dir(self.obj)
        vals = [ str(getattr(self.obj,k)) for k in keys ]
        isdict = [ isinstance(self.obj,dict) for k in keys ]
        has_dict = [ hasattr(self.obj,'__dict__') for k in keys ]
        has_class = [ getattr(self.obj,'__class__') for k in keys ]
        self.items = zip(keys,vals,isdict,has_dict,has_class)
                
                 
    def rowCount(self,parent): 
        return len(self.items) 
 
    def columnCount(self,parent): 
        return len(self.header) 
 
    def data(self,index,role=QtCore.Qt.DisplayRole): 
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.items[index.row()][index.column()]) 
        return QtCore.QVariant() 

    def headerData(self,col,orientation=QtCore.Qt.Horizontal,role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(AttributeModel.header[col])
        return QtCore.QVariant()


class DictModel(QtCore.QAbstractTableModel):
    """A model representing a dictionary."""
    
    header = [ 'key', 'type', 'value' ]
    
    def __init__(self,dic,name,parent=None,*args):
        
        QtCore.QAbstractItemModel.__init__(self,parent,*args) 
        self.dic = dic
        self.name = name
        keys = dic.keys()
        vals = dic.values()
        typs = [ str(type(v)) for v in vals ]
        self.items = zip(keys,typs,vals)
        #print(self.items)
                
    def rowCount(self,parent): 
        return len(self.items) 
 
    def columnCount(self,parent): 
        return len(self.header) 
 
    def data(self,index,role=QtCore.Qt.DisplayRole): 
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.items[index.row()][index.column()]) 
 
        return QtCore.QVariant() 

    def headerData(self,col,orientation=QtCore.Qt.Horizontal,role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(DictModel.header[col])
        return QtCore.QVariant()


class Table(QtGui.QDialog):
    """A dialog widget to show two-dimensional arrays of items."""
    
    def __init__(self,datamodel,caption="pyFormex - Table",parent=None,actions=[('OK',)],default='OK'):
        """Create the Table dialog.
        
        data is a 2-D array of items, mith nrow rows and ncol columns.
        chead is an optional list of ncol column headers.
        rhead is an optional list of nrow row headers.
        """
        if parent is None:
            parent = pf.GUI
        
        QtGui.QDialog.__init__(self,parent)
        self.setWindowTitle(str(caption))
        
        form = QtGui.QVBoxLayout()
        table = QtGui.QTableView()
        table.setModel(datamodel)
        table.horizontalHeader().setVisible(True)
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        #print(table.size())
        form.addWidget(table)

        but = widgets.dialogButtons(self,actions,default)
        form.addLayout(but)
        self.setLayout(form)
        #print(table.size())
        #print(form.size())
        #self.resize(table.size())
        self.table = table
        self.show()
        self.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        #form.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        table.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)




if globals().has_key('tbl'):
    if tbl is not None:
        tbl.close()
tbl = None

def showfields():
    """Show the table of field acronyms."""
    global tbl
    tbl = widgets.Table(result_types.items(),['acronym','description'],actions=[('Cancel',),('Ok',),('Print',tblIndex)])
    tbl.show()
   


def tblIndex():
    print(tbl.table.currentIndex())
    r = tbl.table.currentIndex().row()
    c = tbl.table.currentIndex().column()
    print("(%s,%s)" % (r,c))
    m = tbl.table.model()
    p = m.data(m.index(r,c))
    print(p,p.toString(),p.toBool())

def showattr(name=None,dic=None):
    """Show the table of field acronyms."""
    global tbl
    if dic is None:
        dic = globals()
    k = dic.keys()
    sort(k)
    print(k)
    if name is None:
        name = 'dia_full'
    tbl = AttributeTable(name,dic,actions=[('Cancel',),('Ok',),('Print',tblIndex)])
    tbl.show()

def showdict(dic,name=None):
    global tbl
    model = DictModel(dic,name)
    tbl = Table(model,caption="Dict '%s'" % name,actions=[('Cancel',),('Ok',),('Print',tblIndex)])
    tbl.show()
    tbl.table.resizeColumnsToContents()
    tbl.table.updateGeometry()
    tbl.updateGeometry()


####################


def keys(items):
    """Return the list of keys in items"""
    return [ i[0] for i in items ]

def named_item(items,name):
    """Return the named item"""
    n = keys(items).index(name)
    return items[n]


def showModel(nodes=True,elems=True):
    print DB.elems
    M = [ Mesh(DB.nodes,el,eltype='quad%d'%el.shape[1],prop=i) for i,el in enumerate(DB.elems.itervalues()) ]
    if nodes:
        draw([m.coords for m in M],nolight=True)
    if elems:
        draw(M)
    smoothwire()
    lights(False)
    transparent(True)
    zoomAll()


def showResults(nodes,elems,displ,text,val,showref=False,dscale=100.,
                count=1,sleeptime=-1.):
    """Display a constant or linear field on triangular elements.

    nodes is an array with nodal coordinates
    elems is a single element group or a list of elem groups
    displ are the displacements at the nodes, may be set to None.
    val are the scalar values at the nodes, may also be None.
    If not None, displ should have the same shape as nodes and val
    should have shape (nnodes).

    If dscale is a list of values, the results will be drawn with
    subsequent deformation scales, with a sleeptime intermission,
    and the whole cycle will be repeated count times.
    """
    clear()

    if displ is not None:
    # expand displ if it is smaller than nodes
        # e.g. in 2d returning only 2d displacements
        n = nodes.shape[1] - displ.shape[1]
        if n > 0:
            displ = growAxis(displ,n,axis=1,fill=0.0)

        if nodes.shape != displ.shape:
            warning("The displacements do not match the mesh: the mesh coords have shape %s; but the displacements have shape %s. I will continue without displacements." % (nodes.shape,displ.shape))
            displ = None

    if type(elems) != list:
        elems = [ elems ]

    print ["ELEMS: %s" % str(el.shape) for el in elems ]
    if val is not None:
        print "VAL: %s" % str(val.shape)

    # draw undeformed structure
    if showref:
        ref = [ Mesh(nodes,el,eltype='quad%d'%el.shape[1]) for el in elems ]
        draw(ref,bbox=None,color='green',linewidth=1,mode='wireframe',nolight=True)

    # compute the colors according to the values
    multiplier = 0
    if val is not None:
        if val.shape != (nodes.shape[0],):
            warning("The values do not match the mesh: there are %s nodes in the mesh, and I got values with shape. I will continue without showing values." % (nodes.shape[0],val.shape))
            val = None
            
    if val is not None:
        # create a colorscale and draw the colorlegend
        vmin,vmax = val.min(),val.max()
        if vmin*vmax < 0.0:
            vmid = 0.0
        else:
            vmid = 0.5*(vmin+vmax)

        scalev = [vmin,vmid,vmax]
        print scalev
        logv = [ abs(a) for a in scalev if a != 0.0 ]
        logs = log10(logv)
        logma = int(logs.max())


        if logma < 0:
            multiplier = 3 * ((2 - logma) / 3 )
            #print("MULTIPLIER %s" % multiplier)
            
        CS = ColorScale('RAINBOW',vmin,vmax,vmid,1.,1.)
        cval = array(map(CS.color,val))
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,20,20,30,200,scale=multiplier) 
        pf.canvas.addDecoration(CLA)

    # the supplied text
    if text:
        if multiplier != 0:
            text += ' (* 10**%s)' % -multiplier
        drawText(text,200,30)

    smooth()
    lights(False)
    transparent(False)

    # create the frames while displaying them
    dscale = array(dscale)
    frames = []   # a place to store the drawn frames
    bboxes = []
    if sleeptime >= 0:
        delay(sleeptime)
    for dsc in dscale.flat:

        if displ is None:
            dnodes = nodes
        else:
            dnodes = nodes + dsc * displ
        deformed = [ Mesh(dnodes,el,eltype='quad%d'%el.shape[1]) for el in elems ]
        bboxes.append(bbox(deformed))
        # We store the changing parts of the display, so that we can
        # easily remove/redisplay them
        #print(val)
        if val is None:
            F = [ draw(df,color='blue',view=None,bbox='last',wait=False) for df in deformed ]
        else:
            print [ df.report() + "\nCOLORS %s" % str(cval[el].shape)  for df,el in zip(deformed,elems) ]
            F = [ draw(df,color=cval[el],view=None,bbox='last',wait=False) for df,el in zip(deformed,elems) ]
        T = drawText('Deformation scale = %s' % dsc,200,10)

        # remove the last frame
        # This is a clever trick: we remove the old drawings only after
        # displaying new ones. This makes the animation a lot smoother
        # (though the code is less clear and compact).
        if len(frames) > 0:
            for Fi in frames[-1][0]:
                pf.canvas.removeActor(Fi)
            pf.canvas.removeDecoration(frames[-1][1])
        # add the latest frame to the stored list of frames
        frames.append((F,T))
        wait()

    zoomBbox(bbox(bboxes))
    
    # display the remaining cycles
    count -= 1
    FA,TA = frames[-1]
    while count > 0:
        count -= 1

        for F,T in frames:
            # It would be interesting if addactor would add/remove a list
            # of actors
            for Fi in F:
                pf.canvas.addActor(Fi)
            pf.canvas.addDecoration(T)
            for Fi in FA:
                pf.canvas.removeActor(Fi)
            pf.canvas.removeDecoration(TA)
            pf.canvas.display()
            pf.canvas.update()
            FA,TA = F,T
            wait()


############################# PostProc #################################
## class PostProc(object):
##     """A class to visualize Fe Results."""

##     def __init__(self,DB=None):
##         """Initialize the PostProc. An Fe Results database may be given."""
##         self.resetAll()
##         self.setDB(DB)


##     def resetAll(self):
##         """Reset settings to defaults"""
##         self._show_model = True
##         self._show_elems = True
 
        
##     def postABQ(self,fn=None):
##         """Translate an Abaqus .fil file in a postproc script."""
##         types = [ 'Abaqus results file (*.fil)' ]
##         fn = askFilename(pf.cfg['workdir'],types)
##         if fn:
##             chdir(fn)
##             name,ext = os.path.splitext(fn)
##             post = name+'.post'
##             cmd = "%s/lib/postabq %s > %s" % (pf.cfg['pyformexdir'],fn,post)
##             sta,out = utils.runCommand(cmd)
##             if sta:
##                 pf.message(out)

##     def selectStepInc(self):
##         res = askItems([('Step',self.DB.step,'select',self.DB.res.keys())])
##         if res:
##             step = int(res['Step'])
##             res = askItems([('Increment',None,'select',self.DB.res[step].keys())])
##             if res:
##                 inc = int(res['Increment'])
##         pf.message("Step %s; Increment %s;" % (step,inc))
##         self.DB.setStepInc(step,inc)

    

################### menu #################


## class PostProcGui(PostProc):

##     def __init__(self,*args,**kargs):
##         self.post_button = None
##         self._step_combo = None
##         self._inc_combo = None
##         self.selection = Objects(clas=FeResult)
##         PostProc.__init__(self,*args,**kargs)


##     def setDB(self,DB=None,name=None):
##         """Set the FeResult database.

##         DB can either be an FeResult instance or the name of an exported
##         FeResult.
##         If a name is given, it is displayed on the status bar.
##         """
##         if type(DB) == str:
##             DB = named(DB)
##         if isinstance(DB,FeResult):
##             self.DB = DB
##         else:
##             self.DB = None

##         if self.DB:
## #            self.hideStepInc()
## #            self.hideName()
##             self.showName(name)
##             self.showStepInc()
##         else:
##             self.hideName()
##             self.hideStepInc()


##     def showName(self,name=None):
##         """Show a statusbar button with the name of the DB (hide if None)."""
##         if name is None:
##             self.hideName()
##         else:
##             if self.post_button is None:
##                 self.post_button = widgets.ButtonBox('PostDB:',['None'],[self.select])
##                 pf.GUI.statusbar.addWidget(self.post_button)
##             self.post_button.setText(name)


##     def hideName(self):
##         """Hide the statusbar button with the name of the DB."""
##         if self.post_button:
##             pf.GUI.statusbar.removeWidget(self.post_button)


##     def showStepInc(self):
##         """Show the step/inc combo boxes"""
##         steps = self.DB.getSteps()
##         if steps:
##             self.step_combo = widgets.ComboBox('Step:',steps,self.setStep)
##             pf.GUI.statusbar.addWidget(self.step_combo)
##             self.showInc(steps[0])


##     def showInc(self,step=None):
##         """Show the inc combo boxes"""
##         if step:
##             incs = self.DB.getIncs(step)
##             self.inc_combo = widgets.ComboBox('Inc:',incs,self.setInc)
##             pf.GUI.statusbar.addWidget(self.inc_combo)
    

##     def hideStepInc(self):
##         """Hide the step/inc combo boxes"""
##         if self._inc_combo:
##             pf.GUI.statusbar.removeWidget(self._inc_combo)
##         if self._step_combo:
##             pf.GUI.statusbar.removeWidget(self._step_combo)
             

##     def setStep(self,i):
##         print( "Current index: %s" % i)
##         step = str(self.step_combo.combo.input.currentText())
##         if step != self.DB.step:
##             print("Current step: %s" % step)
##             self.showInc(step)
##             inc = self.DB.getIncs(step)[0]
##             self.setInc(-1)
##             self.DB.setStepInc(step,inc)


##     def setInc(self,i):
##         inc = str(self.inc_combo.combo.input.currentText())
##         if inc != self.DB.inc:
##             self.DB.setStepInc(step,inc)
##         print("Current step/inc: %s/%s" % (self.DB.step,self.DB.inc))
        

##     def select(self,sel=None):
##         sel = self.selection.ask1()
##         if sel:
##             self.setDB(sel,self.selection[0])


########## Postproc results dialog #######

result_types = ODict([
    ('','None'),
    ('U','[Displacement]'),
    ('U0','X-Displacement'),
    ('U1','Y-Displacement'),
    ('U2','Z-Displacement'),
    ('S0','X-Normal Stress'),
    ('S1','Y-Normal Stress'),
    ('S2','Z-Normal Stress'),
    ('S3','XY-Shear Stress'),
    ('S4','XZ-Shear Stress'),
    ('S5','YZ-Shear Stress'),
    ('SP0','1-Principal Stress'),
    ('SP1','2-Principal Stress'),
    ('SP2','3-Principal Stress'),
    ('SF0','x-Normal Membrane Force'),
    ('SF1','y-Normal Membrane Force'),
    ('SF2','xy-Shear Membrane Force'),
    ('SF3','x-Bending Moment'),
    ('SF4','y-Bending Moment'),
    ('SF5','xy-Twisting Moment'),
    ('SINV0','Mises Stress'),
    ('SINV1','Tresca Stress'),
    ('SINV2','Hydrostatic Pressure'),
    ('SINV6','Third Invariant'),
    ('COORD0','X-Coordinate'),
    ('COORD1','Y-Coordinate'),
    ('COORD2','Z-Coordinate'),
    ('Computed','Distance from a point'),
    ])


input_items = [
    dict(name='feresult',text='FE Result DB',value='',itemtype='info'),
    dict(name='elgroup',text='Element Group',choices=['--ALL--',]),
    dict(name='restype',text='Type of result',choices=result_types.values()),
    dict(name='loadcase',text='Load case',value=0),
    dict(name='autoscale',text='Autocalculate deformation scale',value=True),
    dict(name='dscale',text='Deformation scale',value=100.),
    dict(name='showref',text='Show undeformed configuration',value=True),
    dict(name='animate',text='Animate results',value=False),
    dict(name='shape',text='Amplitude shape',value='linear',itemtype='radio',choices=['linear','sine']),
    dict(name='cycle',text='Animation cycle',value='updown',itemtype='radio',choices=['up','updown','revert']),
    dict(name='count',text='Number of cycles',value=5),
    dict(name='nframes',text='Number of frames',value=10),
    dict(name='sleeptime',text='Animation sleeptime',value=0.1), 
    ]


selection = Objects(clas=FeResult)
dialog = None
DB = None


def show():
    """Show the results"""
    data = dialog_getresults()
    #print(data)
    globals().update(data)
    nodes = DB.nodes
    if elgroup == '--ALL--':
        elems = DB.elems.values()
    else:
        elems = [ DB.elems[elgroup] ]

    dscale = data['dscale']
    displ = DB.getres('U')
    if displ is not None:
        displ = displ[:,0:3]

        if autoscale:
            siz0 = Coords(nodes).sizes()
            siz1 = Coords(displ).sizes()
            w = where(siz0 > 0.0)[0]
            dscale = niceNumber(0.5/(siz1[w]/siz0[w]).max())

    if animate:
        dscale = dscale * frameScale(nframes,cycle=cycle,shape=shape) 

    # Get the scalar element result values from the results.
    txt = 'No Results'
    val = None
    if resindex > 0:
        key = result_types.keys()[resindex]
        print("RESULT KEY = %s" % key)
        if key == 'Computed':
            if askPoint():
                val = Coords(nodes).distanceFromPoint(point)
        else:
            val = DB.getres(key)
            if key == 'U':
                val = norm2(val)
    if val is not None:
        txt = result_types.values()[resindex]
    showResults(nodes,elems,displ,txt,val,showref,dscale,count,sleeptime)
    return val


def setDB(db):
    """Set the current result. db is an FeResult instance."""
    global DB
    if isinstance(db,FeResult):
        DB = db
    else:
        DB = None
    pf.PF['PostProcMenu_result'] = DB

    
def selectDB(db=None):
    """Select the result database to work upon.

    If db is an FeResult instance, it is set as the current database.
    If None is given, a dialog is popped up to select one.

    If a database is succesfully selected, the screen is cleared and the
    geometry of the model is displayed.
    
    Returns the database or None.
    """
    if not isinstance(db,FeResult):
        db = selection.ask1()
        if db:
            print("Selected results database: %s" % selection.names[0])
    if db:
        setDB(db)
        clear()
        print(DB.about.get('heading','No Heading'))
        print('Stress tensor has %s components' % DB.datasize['S'])
        showModel()
    return db

    
def importFlavia(fn=None):
    """Import a flavia file and select it as the current results.

    Flavia files are the postprocessing format used by GiD pre- and
    postprocessor, and can also be written by the FE program calix.
    There usually are two files named 'BASE.flavia.msh' and 'BASE.flavia.res'
    which hold the FE mesh and results, respectively.
    
    This functions asks the user to select a flavia file (either mesh or
    results), will then read both the mesh and corrseponding results files,
    and store the results in a FeResult instance, which will be set as the
    current results database for the postprocessing menu.
    """
    from plugins.flavia import readFlavia
    if fn is None:
        types = [ utils.fileDescription('flavia'), utils.fileDescription('all') ]
        fn = askFilename(pf.cfg['workdir'],types)
    if fn:
        chdir(fn)
        if fn.endswith('.msh'):
            meshfile = fn
            resfile = utils.changeExt(fn,'res')
        else:
            resfile = fn
            meshfile = utils.changeExt(fn,'msh')
            
        db = readFlavia(meshfile,resfile)
        if not isinstance(db,FeResult):
            warning("!Something went wrong during the import of the flavia database %s" % fn)
            return

        ### ok: export and select the DB
        name = os.path.splitext(os.path.basename(fn))[0].replace('.flavia','')
        export({name:db})
        db.printSteps()
        print db.R
        print db.datasize
        
        selection.set([name])
        selectDB(db)
        

    
def importDB(fn=None):
    """Import a _post.py database and select it as the current."""
    
    if fn is None:
        types = utils.fileDescription('postproc')
        fn = askFilename(pf.cfg['workdir'],types)
    if fn:
        chdir(fn)
        size = os.stat(fn).st_size
        if size > 1000000 and ask("""
BEWARE!!!

The size of this file is very large: %s bytes
It is unlikely that I will be able to process this file.
I strongly recommend you to cancel the operation now.
""" % size,["Continue","Cancel"]) != "Continue":
            return

        # import the results DB
        play(fn)

        ### check whether the import succeeded
        name = FeResult._name_
        db = pf.PF[name]
        if not isinstance(db,FeResult):
            warning("!Something went wrong during the import of the database %s" % fn)
            return
        
        ### ok: select the DB
        selection.set([name])
        selectDB(db)

    
def checkDB():
    """Make sure that a database is selected.

    If no results database was already selected, asks the user to do so.
    Returns True if a databases is selected.
    """
    print(DB)
    if not isinstance(DB,FeResult):
        selectDB()
    return isinstance(DB,FeResult)


def dialog_getresults():
    """Return the dialog data with short keys."""
    dialog.acceptData()
    data = dialog.results
    data['resindex'] = result_types.values().index(data['restype'])
    return data


def dialog_reset(data=None):
    # data is a dict with short keys/data
    if data is None:
        data = dict((i['name'],i.get('value',None)) for i in input_items)
    dialog.updateData(data)


def open_dialog():
    global dialog
    if not checkDB():
        warning("No results database was selected!")
        return
    close_dialog()

    actions = [
        ('Close',close_dialog),
        ('Reset',reset),
        # ('Select DB',selectDB),
        ('Show',show),
        # ('Show Fields',showfields),
        # ('Show Attr',showattr),
        ]
    dialog = widgets.InputDialog(input_items,caption='Results Dialog',actions=actions,default='Show')
    # Update the data items from saved values
    try:
        newdata = named('PostProcMenu_data')
        print newdata
    except:
        newdata = {}
        pass
    if selection.check(single=True):
        newdata['feresult'] = selection.names[0]
    if DB:
        newdata['elgroup'] = ['--ALL--',] + DB.elems.keys()
    dialog.updateData(newdata)
    dialog.show()
    #pf.PF['__PostProcMenu_dialog__'] = dialog


def close_dialog():
    global dialog
    if dialog:
        pf.PF['PostProcMenu_data'] = dialog_getresults()
        dialog.close()
        dialog = None
    if tbl:
        tbl.close()


def askPoint():
    global point
    res = askItems([('Point',point)])
    if res:
        point = res['Point']
        return point
    else:
        return None


################################## Menu #############################

_menu = 'Postproc'

def create_menu():
    """Create the Postproc menu."""
    MenuData = [
#        ("&Translate Abaqus .fil to FeResult database",P.postABQ),
        ("&Read FeResult Database",importDB),
        ("&Read Flavia Database",importFlavia),
        ("&Select FeResult Data",selectDB),
#        ("&Forget FeResult Data",P.selection.forget),
        ("---",None),
        ("Show Geometry",showModel),
#        ("Select Step/Inc",P.selectStepInc),
#        ("Show Results",P.postProc),
        ("Results Dialog",open_dialog),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close menu",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help')


def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the menu."""
    pf.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the Postproc menu."""
    global DB
    close_menu()
    DB = pf.PF.get('PostProcMenu_result',None)
    print("Current database %s" % DB)
    import plugins
    plugins.refresh('postproc_menu')
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    reload_menu()


# End

