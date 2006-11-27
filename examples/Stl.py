#!/usr/bin/env pyformex
# $Id: Stl.py 156 2006-11-06 19:14:25Z bverheg $

"""Stl.py

All this script does when executed is to construct a specialized menu with
actions defined in this script, and display the menu.
Then the script bails out, leaving the user the option to use the menu.
Selecting an option removes the menu, so each action should redraw it.
"""

from plugins import f2abq, stl, tetgen, stl_abq
from gui import widgets
import commands, os

clear()

global project,F,nodes,elems,surf
project = F = nodes = elems = surf = None

# Actions

def new_project():
    global project
    """Set a new project name by asking for an .stl filename."""
    message("Set the project name by choosing the input .stl file\n(Hint: You can run the Sphere_stl example to create one.)")
    fn = askFilename(GD.cfg['workdir'],"Stl files (*.stl)")
    if fn:
        os.chdir(os.path.dirname(fn))
        message("Your current workdir is %s" % os.getcwd())
        project = os.path.splitext(fn)[0]

def read_stl():
    global project,F
    """Read the .stl surface model into a Formex."""
    if project is None:
        new_project()
    if project:
        fn = project+'.stl'
        message("Reading file %s" % fn)
        F = Formex(stl.read_ascii(fn))
        message("There are %d triangles in the model" % F.f.shape[0])
        message("The bounding box is\n%s" % F.bbox())
        show_stl()

def show_stl():
    """Display the .stl model."""
    global F
    if F:
        updateGUI()
        GD.gui.update()
        GD.app.processEvents()
        linewidth(1)
        draw(F,color='green')

def save_stl():
    """Save the stl model."""
    global project,F
    if F is NOne:
        return
    fn = askFilename(GD.cfg['workdir'],"Stl files (*.stl)",exist=False)
    if fn:
        os.chdir(os.path.dirname(fn))
        message("Your current workdir is %s" % os.getcwd())
        project = os.path.splitext(fn)[0]
        if not os.path.exists(fn) or ack("File %s already exists. Overwrite?" % fn):
            stl.write_ascii(F.f,fn)

def center_stl():
    """Center the stl model."""
    global F
    updateGUI()
    center = array(F.center())
    print center
    clear()
    draw(F,color='yellow',wait=False)
    F = F.translate(-center)
    draw(F,color='green')

def scale_stl():
    """Scale the stl model."""
    global F
    itemlist = [ [ 'X-scale',1.0], [ 'Y-scale',1.0], [ 'Z-scale',1.0] ] 
    res,accept = widgets.inputDialog(itemlist,'Scaling Parameters').process()
    if accept:
        print res
        updateGUI()
        clear()
        draw(F,color='yellow',wait=False)
        F = F.scale(map(float,[r[1] for r in res]))
        draw(F,color='green')

def rotate_stl():
    """Rotate the stl model."""
    global F
    itemlist = [ [ 'axis',0], ['angle','0.0'] ] 
    res,accept = widgets.inputDialog(itemlist,'Rotation Parameters').process()
    if accept:
        print res
        updateGUI()
        clear()
        draw(F,color='yellow',wait=False)
        F = F.rotate(float(res[1][1]),int(res[0][1]))
        draw(F,color='green')
        
def clip_stl():
    """Clip the stl model."""
    global F
    itemlist = [['axis',0],['begin',0.0],['end',1.0]]
    res,accept = widgets.inputDialog(itemlist,'Clipping Parameters').process()
    if accept:
        print res
        updateGUI()
        clear()
        bb = F.bbox()
        print "Original bbox: %s" % bb 
        xmi = bb[0][0]
        xma = bb[1][0]
        dx = xma-xmi
        axis = int(res[0][1])
        xc1 = xmi + float(res[1][1]) * dx
        xc2 = xmi + float(res[2][1]) * dx
        w = F.where(dir=axis,xmin=xc1,xmax=xc2)
        oldF = F
        draw(F.cclip(w),color='yellow',wait=False)
        F = F.clip(w)
        message("Bounding box = %s" % F.bbox())
        linewidth(2)
        draw(F,color='green')

def section_stl():
    """Sectionize the stl model."""
    global F,sect,ctr,diam
    clear()
    linewidth(1)
    draw(F,color='yellow')
    bb = F.bbox()
    message("Bounding box = %s" % bb)

    itemlist = [['number of sections',20],['relative thickness',0.1]]
    res,accept = widgets.inputDialog(itemlist,'Sectioning Parameters').process()
    sect = []
    ctr = []
    diam = []
    if accept:
        n = int(res[0][1])
        th = float(res[1][1])
        xmin = bb[0][0]
        xmax = bb[1][0]
        dx = (xmax-xmin) / n
        dxx = dx * th
        X = xmin + arange(n+1) * dx
        message("Sections are taken at X-values: %s" % X)

        c = zeros([n,3],float)
        d = zeros([n,1],float)
        linewidth(2)

        for i in range(n+1):
            G = F.clip(F.where(dir=0,xmin=X[i]-dxx,xmax=X[i]+dxx))
            draw(G,color='blue',view=None)
            GD.canvas.update()
            C = G.center()
            H = Formex(G.f-C)
            x,y,z = H.x(),H.y(),H.z()
            D = 2 * (x*x+y*y+z*z).mean()
            message("Section Center: %s; Diameter: %s" % (C,D))
            sect.append(G)
            ctr.append(C)
            diam.append(D)


def flytru_stl():
    """Fly through the stl model."""
    global ctr
    print ctr
    fc = array(ctr).reshape((-1,1,3))
    Fc = Formex(fc)
    path = connect([Fc,Fc])
    print path.shape()
    flyAlong(path)
    

def export_stl():
    """Export an stl model stored in Formex F in Abaqus .inp format."""
    global project,F
    if ack("Creating nodes and elements.\nFor a large model, this could take quite some time!"):
        GD.app.processEvents()
        message("Creating nodes and elements.")
        nodes,elems = F.nodesAndElements()
        nnodes = nodes.shape[0]
        nelems = elems.shape[0]
        message("There are %d unique nodes and %d triangle elements in the model." % (nnodes,nelems))
        stl_abq.abq_export(project+'.inp',nodes,elems,'S3',"Created by stl_examples.py")
#    menu.process()
    

def create_tetgen():
    """Generate a volume tetraeder mesh inside an stl surface."""
    fn = project + '.stl'
    if os.path.exists(fn):
        sta,out = commands.getstatusoutput('tetgen %s' % fn)
        message(out)
#    menu.process()


def read_tetgen(surface=True, volume=True):
    """Read a tetgen model from files  fn.node, fn.ele, fn.smesh."""
    global nodes,elems,surf
    if project is None:
        new_project()
    if project:
        nodes = tetgen.readNodes(project+'.1.node')
        print "Read %d nodes" % nodes.shape[0]
        if volume:
            elems = tetgen.readElems(project+'.1.ele')
            print "Read %d tetraeders" % elems.shape[0]
        if surface:
            surf = tetgen.readSurface(project+'.1.smesh')
            print "Read %d triangles" % surf.shape[0]
#    menu.process()


def read_tetgen_surface():
    read_tetgen(volume=False)

    
def read_tetgen_volume():
    read_tetgen(surface=False)


def show_tetgen_surface():
    global nodes,surf
    updateGUI()
    if surf is not None:
        surface = Formex(nodes[surf-1])
        clear()
        draw(surface,color='red')


def show_tetgen_volume():
    global nodes,elems
    updateGUI()
    if elems is not None:
        volume = Formex(nodes[elems-1])
        clear()
        draw(volume,color='random',eltype='tet')


def export_tetgen_surface():
    global nodes,surf
    updateGUI()
    if surf is not None:
        print "Exporting surface model"
        stl_abq.abq_export('%s-surface.inp' % project,nodes,surf,'S3',"Abaqus model generated by tetgen from surface in STL file %s.stl" % project)



def export_tetgen_volume():
    global nodes,elems
    updateGUI()
    if elems is not None:
        print "Exporting volume model"
        stl_abq.abq_export('%s-volume.inp' % project,nodes,elems,'C3D%d' % elems.shape[1],"Abaqus model generated by tetgen from surface in STL file %s.stl" % project)


# Menu
menu = widgets.Menu('STL') # Should be done before defining MenuData!

MenuData = [
    ("Action","&New project",new_project),
    ("Action","&Read .stl file",read_stl),
    ("Action","&Show .stl model",show_stl),
    ("Action","&Center .stl model",center_stl),
    ("Action","&Rotate .stl model",rotate_stl),
    ("Action","&Clip .stl model",clip_stl),
    ("Action","&Scale .stl model",scale_stl),
    ("Action","&Save .stl model",save_stl),
    ("Action","&Sectionize .stl model",section_stl),
    ("Action","&Fly .stl model",flytru_stl),
    ("Action","&Export .stl model to Abaqus (SLOW!)",export_stl),
    ("Action","&Create tetgen model",create_tetgen),
    ("Action","&Read tetgen surface",read_tetgen_surface),
    ("Action","&Read tetgen volume",read_tetgen_volume),
    ("Action","&Read tetgen surface+volume",read_tetgen),
    ("Action","&Show tetgen surface",show_tetgen_surface),
    ("Action","&Show tetgen volume",show_tetgen_volume),
    ("Action","&Export surface to Abaqus",export_tetgen_surface),
    ("Action","&Export volume to Abaqus",export_tetgen_volume),
    ("Action","&Close Menu",menu.close),
    ]

for key,txt,val in MenuData:
    menu.addItem(txt,val)

# End
