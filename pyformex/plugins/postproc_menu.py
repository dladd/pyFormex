#!/usr/bin/env pyformex --gui
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

import pyformex as GD
from plugins.postproc import *
from plugins.fe_post import FeResult
from plugins.objects import Objects
from gui.colorscale import ColorScale,ColorLegend
from gui import decors,canvas,widgets
import utils
from odict import ODict

import commands

from numpy import *
from formex import *
from gui.draw import *
from gui.colors import *


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
        print self.items
                
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
            parent = GD.GUI
        
        QtGui.QDialog.__init__(self,parent)
        self.setWindowTitle(str(caption))
        
        form = QtGui.QVBoxLayout()
        table = QtGui.QTableView()
        table.setModel(datamodel)
        table.horizontalHeader().setVisible(True)
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        print table.size()
        form.addWidget(table)

        but = widgets.dialogButtons(self,actions,default)
        form.addLayout(but)
        self.setLayout(form)
        print table.size()
        #print form.size()
        #self.resize(table.size())
        self.table = table
        self.show()
        self.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        #form.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        table.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)




if globals().has_key('tbl'):
    tbl.close()
tbl = None

def showfields():
    """Show the table of field acronyms."""
    global tbl
    tbl = widgets.Table(res_types,['acronym','description'],actions=[('Cancel',),('Ok',),('Print',tblIndex)])
    tbl.show()
   


def tblIndex():
    print tbl.table.currentIndex()
    r = tbl.table.currentIndex().row()
    c = tbl.table.currentIndex().column()
    print "(%s,%s)" % (r,c)
    m = tbl.table.model()
    p = m.data(m.index(r,c))
    print p,p.toString(),p.toBool()

def showattr(name=None,dic=None):
    """Show the table of field acronyms."""
    global tbl
    if dic is None:
        dic = globals()
    k = dic.keys()
    sort(k)
    print k
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
    if nodes:
        Fn = Formex(DB.nodes)
        draw(Fn)
    if elems:
        Fe = [ Formex(DB.nodes[e],i+1) for i,e in enumerate(DB.elems.itervalues()) ]
        draw(Fe)
    smoothwire()
    lights(False)
    transparent(True)
    zoomAll()


def showResults(nodes,elems,displ,text,val,showref=False,dscale=100.,
                count=1,sleeptime=-1.):
    """Display a constant or linear field on triangular elements.

    nodes is an array with nodal coordinates
    elems is a single element group or a list of elem groups
    displ are the displacements at the nodes
    val are the scalar values at the nodes

    If dscale is a list of values, the results will be drawn with
    subsequent deformation scales, with a sleeptime intermission,
    and the whole cycle will be repeated count times.
    """
    clear()

    if type(elems) != list:
        elems = [ elems ]

    # draw undeformed structure
    if showref:
        ref = [ Formex(nodes[el]) for el in elems ]
        draw(ref,bbox=None,color='green',linewidth=1,mode='wireframe')

    # compute the colors according to the values
    multiplier = 0
    if val is not None:
        # create a colorscale and draw the colorlegend
        vmin,vmax = val.min(),val.max()
        if vmin*vmax < 0.0:
            vmid = 0.0
        else:
            vmid = 0.5*(vmin+vmax)

        scalev = [vmin,vmid,vmax]
        logv = [ a for a in scalev if a != 0.0 ]
        logs = log10(logv)
        logma = int(logs.max())


        if logma < 0:
            multiplier = 3 * ((2 - logma) / 3 )
            print "MULTIPLIER %s" % multiplier
            
        CS = ColorScale([blue,green,red],vmin,vmax,vmid,1.,1.)
        cval = array(map(CS.color,val))
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,10,20,30,200,scale=multiplier) 
        GD.canvas.addDecoration(CLA)

    # the supplied text
    if text:
        if multiplier != 0:
            text += ' (* 10**%s)' % -multiplier
        drawtext(text,150,30,'tr18')

    smooth()
    lights(False)
    transparent(True)

    # create the frames while displaying them
    dscale = array(dscale)
    frames = []   # a place to store the drawn frames
    bbox = []
    for dsc in dscale.flat:

        print dsc
        #print nodes.shape
        #print displ.shape
        dnodes = nodes + dsc * displ
        deformed = [ Formex(dnodes[el]) for el in elems ]
        bboxes = [ df.bbox() for df in deformed ]
        bbox.append(Formex(bboxes).bbox())
        # We store the changing parts of the display, so that we can
        # easily remove/redisplay them
        print val
        if val is None:
            F = [ draw(df,color='blue',view='__last__',bbox=None,wait=None) for df in deformed ]
        else:
            F = [ draw(df,color=cval[el],view='__last__',bbox=None,wait=None) for df,el in zip(deformed,elems) ]
        T = drawtext('Deformation scale = %s' % dsc,150,10,'tr18')

        # remove the last frame
        # This is a clever trick: we remove the old drawings only after
        # displaying new ones. This makes the animation a lot smoother
        # (though the code is less clear and compact).
        if len(frames) > 0:
            for Fi in frames[-1][0]:
                GD.canvas.removeActor(Fi)
            GD.canvas.removeDecoration(frames[-1][1])
        # add the latest frame to the stored list of frames
        frames.append((F,T))
        if sleeptime > 0.:
            GD.app.processEvents()
            sleep(sleeptime)

    zoomBbox(Formex(bbox).bbox())
    
    # display the remaining cycles
    count -= 1
    FA,TA = frames[-1]
    while count > 0:
        count -= 1

        for F,T in frames:
            # It would be interesting if addactor would add/remove a list
            # of actors
            for Fi in F:
                GD.canvas.addActor(Fi)
            GD.canvas.addDecoration(T)
            for Fi in FA:
                GD.canvas.removeActor(Fi)
            GD.canvas.removeDecoration(TA)
            GD.canvas.display()
            GD.canvas.update()
            FA,TA = F,T
            if sleeptime > 0.:
                GD.app.processEvents()
                sleep(sleeptime)


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
##         fn = askFilename(GD.cfg['workdir'],types,exist=True)
##         if fn:
##             chdir(fn)
##             name,ext = os.path.splitext(fn)
##             post = name+'.post'
##             cmd = "%s/lib/postabq %s > %s" % (GD.cfg['pyformexdir'],fn,post)
##             sta,out = utils.runCommand(cmd)
##             if sta:
##                 GD.message(out)


##     def importDB(self,fn=None):
##         if fn is None:
##             types = utils.fileDescription('postproc')
##             fn = askFilename(GD.cfg['workdir'],types,exist=True)
##         if fn:
##             chdir(fn)
##             ###
##             ### Warning for obsolete feature
##             ### Will be removed in version 0.8
##             if fn.endswith('_post.py'):
##                 ans = ask("The '_post.py' extension for postprocessing databases is obsolete and should be avoided. Use the '.post' extension instead.\n\nDo you want to rename the database now?",['Keep','Rename','Cancel'])
##                 if ans == 'Cancel':
##                     return
##                 elif ans == 'Rename':
##                     newfn = fn.replace('_post.py','.post')
##                     while os.path.exists(newfn):
##                         newfn = newfn.replace('.post','_1.post')
##                     os.rename(fn,newfn)
##                     fn = newfn

##             size = os.stat(fn).st_size
##             if size > 1000000 and ask("""
##     BEWARE!!!

##     The size of this file is very large: %s bytes
##     It is unlikely that I will be able to process this file.
##     I strongly recommend you to cancel the operation now.
##     """ % size,["Continue","Cancel"]) != "Continue":
##                 return

##             project = os.path.basename(os.path.splitext(fn)[0])
##             #### Currenty, the postabq always uses the global 'DB'
##             ##DB = FeResult()
##             export({'DB':self.DB})
##             play(fn)
##             #### We export it under the project name
##             export({project:GD.PF['DB']})
##             #### and delete the 'DB' name
##             del GD.PF['DB']
##             ### now select the DB
##             self.setDB(GD.PF[project],project)
##             GD.message(self.DB.about['heading'])
##             self.DB.printSteps()
##             #self.showModel()


##     def selectStepInc(self):
##         res = askItems([('Step',self.DB.step,'select',self.DB.res.keys())])
##         if res:
##             step = int(res['Step'])
##             res = askItems([('Increment',None,'select',self.DB.res[step].keys())])
##             if res:
##                 inc = int(res['Increment'])
##         GD.message("Step %s; Increment %s;" % (step,inc))
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
##                 GD.GUI.statusbar.addWidget(self.post_button)
##             self.post_button.setText(name)


##     def hideName(self):
##         """Hide the statusbar button with the name of the DB."""
##         if self.post_button:
##             GD.GUI.statusbar.removeWidget(self.post_button)


##     def showStepInc(self):
##         """Show the step/inc combo boxes"""
##         steps = self.DB.getSteps()
##         if steps:
##             self.step_combo = widgets.ComboBox('Step:',steps,self.setStep)
##             GD.GUI.statusbar.addWidget(self.step_combo)
##             self.showInc(steps[0])


##     def showInc(self,step=None):
##         """Show the inc combo boxes"""
##         if step:
##             incs = self.DB.getIncs(step)
##             self.inc_combo = widgets.ComboBox('Inc:',incs,self.setInc)
##             GD.GUI.statusbar.addWidget(self.inc_combo)
    

##     def hideStepInc(self):
##         """Hide the step/inc combo boxes"""
##         if self._inc_combo:
##             GD.GUI.statusbar.removeWidget(self._inc_combo)
##         if self._step_combo:
##             GD.GUI.statusbar.removeWidget(self._step_combo)
             

##     def setStep(self,i):
##         print  "Current index: %s" % i
##         step = str(self.step_combo.combo.input.currentText())
##         if step != self.DB.step:
##             print "Current step: %s" % step
##             self.showInc(step)
##             inc = self.DB.getIncs(step)[0]
##             self.setInc(-1)
##             self.DB.setStepInc(step,inc)


##     def setInc(self,i):
##         inc = str(self.inc_combo.combo.input.currentText())
##         if inc != self.DB.inc:
##             self.DB.setStepInc(step,inc)
##         print "Current step/inc: %s/%s" % (self.DB.step,self.DB.inc)
        

##     def select(self,sel=None):
##         sel = self.selection.ask1()
##         if sel:
##             self.setDB(sel,self.selection[0])


########## Postproc results dialog #######

res_types = [
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
    ]

res_dict = ODict(res_types)

dia_full = [
    ['feresult','FE Result DB','','info'],
    ['elgroup','Element Group',None,'select',['--ALL--',]],
    ['resindex','Type of result',None,'select',res_dict.values()],
    ['loadcase','Load case',0],
    ['autoscale','Autocalculate deformation scale',True],
    ['dscale','Deformation scale',100.],
    ['showref','Show undeformed configuration',True],
    ['animate','Animate results',False],
    ['shape','Amplitude shape','linear','radio',['linear','sine']],
    ['cycle','Animation cycle','updown','radio',['up','updown','revert']],
    ['count','Number of cycles',5],
    ['nframes','Number of frames',10],
    ['sleeptime','Animation sleeptime',0.1], 
    ]


new_vals = dict(
    feresult = 'some result DB',
    elgroup = '',
    resindex = 'Y-Displacement',
    loadcase = 1,
    autoscale = False,
    dscale = 35.5,
    showref = True,
    animate = True,
    shape = 'sine',
    cycle = 'revert',
    count = 3,
    nframes = 999,
    sleeptime = 1,
    )

dia_dict = ODict([ (c[0],c[1:]) for c in dia_full ])
dia_defaults = dict([ (c[0],c[2]) for c in dia_full ])

selection = Objects(clas=FeResult)
dialog = None
DB = None


def show():
    """Show the results"""
    data = dialog_getdata()
    print data
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
        key = res_dict.keys()[resindex]
        print "RESULT KEY = %s" % key
        if key == 'Computed':
            if askPoint():
                val = Coords(nodes).distanceFromPoint(point)
        else:
            val = DB.getres(key)
            if key == 'U':
                val = norm2(val)
    if val is not None:
        txt = res_dict.values()[resindex]
    print nodes.shape
    print displ.shape
    showResults(nodes,elems,displ,txt,val,showref,dscale,count,sleeptime)
    return val


def newvals():
    reset(new_vals)


def dialog_getdata():
    """Return the dialog data with short keys."""
    dialog.acceptData()
    data = dialog.result
    data = dict([ (c[0],data[c[1]]) for c in dia_full ])
    data['resindex'] = res_dict.values().index(data['resindex'])
    return data

    
def dialog_update(data):
    # data is a dict with short keys/data
    vals = {}
    for v in data:
        d = dia_dict[v]
        vals[d[0]] = data[v]
    print vals
    dialog.updateData(vals)


def reset(data=None):
    # data is a dict with short keys/data
    if data is None:
        data = dia_defaults
    GD.PF['__PostProcMenu_data__'] = data
    dialog_update(data)


def setDB(db):
    """Set the current result. db is an FeResult instance."""
    global DB
    if isinstance(db,FeResult):
        DB = db
    else:
        DB = None
    GD.PF['__PostProcMenu_result__'] = DB

    
def selectDB():
    """Select the current result database.

    If db is an FeResult instance, it is set as the current database.
    If None is given, a dialog is popped up to select one.
    Returns the global DB variable (set to the selected database or None).
    """
    db = selection.ask1()
    print "Selected results database: %s" % selection.names
    if db:
        setDB(db)
        showModel()
        print 'Stress tensor has %s components' % DB.data_size['S']
    
    
def checkDB():
    """Make sure that a database is selected.

    If no results database was already selected, asks the user to do so.
    Returns True if a databases is selected.
    """
    print DB
    if not isinstance(DB,FeResult):
        selectDB()
    return isinstance(DB,FeResult)


def open_results_dialog():
    global dialog
    if not checkDB():
        warning("No results database was selected!")
        return
    data = GD.PF.get('__PostProcMenu_data__',dia_defaults)
    print "SAVED DATA",data
    for k,v in data.items():
        dia_dict[k][1] = v
    if selection.check(single=True):
        dia_dict['feresult'][1] = selection.names[0]
    if DB:
        dia_dict['elgroup'][3] = ['--ALL--',] + DB.elems.keys()
    actions = [('Close',close_dialog),
               ('Reset',reset),
#               ('Select DB',selectDB),
               ('Show',show),
#               ('NewVals',newvals),
#               ('Show Fields',showfields),
#               ('Show Attr',showattr),
               ]
    dialog = widgets.InputDialog(dia_dict.values(),caption='Results Dialog',actions=actions,default='Show')
    dialog.show()
    GD.PF['__PostProcMenu_dialog__'] = dialog


def close_dialog():
    global dialog
    if dialog:
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


def create_menu():
    """Create the Postproc menu."""
    MenuData = [
#        ("&Translate Abaqus .fil to FeResult database",P.postABQ),
#        ("&Open FeResult Database",P.importDB),
        ("&Select FeResult Data",selectDB),
#        ("&Forget FeResult Data",P.selection.forget),
        ("---",None),
        ("Show Geometry",showModel),
#        ("Select Step/Inc",P.selectStepInc),
#        ("Show Results",P.postProc),
        ("Results Dialog",open_results_dialog),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close menu",close_menu),
        ]
    return widgets.Menu('Postproc',items=MenuData,parent=GD.GUI.menu,before='help')
  
def show_menu():
    """Show the Postproc menu."""
    if not GD.GUI.menu.item('Postproc'):
        create_menu()
        
def close_menu():
    """Close the Postproc menu."""
    m = GD.GUI.menu.item('Postproc')
    if m :
        m.remove()
        #GD.GUI.statusbar.removeWidget(GD.GUI.postbutton)

def reload_menu():
    """Reload the Postproc menu."""
    global DB
    close_menu()
    DB =  GD.PF.get('__PostProcMenu_result__',None)
    print "Current database %s" % DB
    show_menu()


if __name__ == "draw":
    reload_menu()
    
elif __name__ == "__main__":
    print __doc__

# End

