#!/usr/bin/env pyformex
# $Id$

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
project = F = nodes = elems = surf =None

# Actions

def read_stl():
    global project,F
    """Read an .stl surface model into a Formex."""
    message("Choose the input .stl file\nIf you have none, run the Sphere_stl example to create one.")
    fn = askFilename(GD.cfg['workdir'],"Stl files (*.stl)")
    if fn:
        os.chdir(os.path.dirname(fn))
        message("Your current workdir is %s" % os.getcwd())
        project = os.path.splitext(fn)[0]
        message("Reading file %s" % fn)
        F = Formex(stl.read_ascii(fn))
        message("There are %d triangles in the model" % F.f.shape[0])
        message("The bounding box is\n%s" % F.bbox())
    menu.process()

def show_stl():
    """Display the .stl model."""
    global F
    draw(F,color='green')
    menu.process()


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
    menu.process()
    

def create_tetgen():
    """Generate a volume tetraeder mesh inside an stl surface."""
    fn = project + '.stl'
    if os.path.exists(fn):
        sta,out = commands.getstatusoutput('tetgen %s' % fn)
        message(out)
    menu.process()


def read_tetgen(surface=True, volume=True):
    """Read a tetgen model from files  fn.node, fn.ele, fn.smesh."""
    global nodes,elems,surf
    nodes = tetgen.readNodes(project+'.1.node')
    print "Read %d nodes" % nodes.shape[0]
    if volume:
        elems = tetgen.readElems(project+'.1.ele')
        print "Read %d tetraeders" % elems.shape[0]
    if surface:
        surf = tetgen.readSurface(project+'.1.smesh')
        print "Read %d triangles" % surf.shape[0]
    menu.process()


def read_tetgen_surface():
    read_tetgen(volume=False)

    
def read_tetgen_volume():
    read_tetgen(surface=False)


def show_tetgen_surface():
    global nodes,elems,surf
    if surf is not None:
        surface = Formex(nodes[surf-1])
        clear()
        draw(surface,color='red')
    menu.process()
    

def show_tetgen_volume():
    global nodes,elems,surf
    if elems is not None:
        volume = Formex(nodes[elems-1])
        clear()
        draw(volume,color='random')
    menu.process()
    

# Menu
menu = widgets.Menu('STL',True) # Should be done before defining MenuData!

MenuData = [
    ("Action","&Read .stl ",read_stl),
    ("Action","&Show .stl ",show_stl),
    ("Action","&Export .stl ",export_stl),
    ("Action","&Create tetgen",create_tetgen),
    ("Action","&Read tetgen surface",read_tetgen_surface),
    ("Action","&Read tetgen volume",read_tetgen_volume),
    ("Action","&Read tetgen surface+volume",read_tetgen),
    ("Action","&Show tetgen surface",show_tetgen_surface),
    ("Action","&Show tetgen volume",show_tetgen_volume),
    ("Action","&Done",menu.close),
    ]

for key,txt,val in MenuData:
    print type(val)
    menu.addItem(txt,val)

# show the menu
menu.process()

# End
