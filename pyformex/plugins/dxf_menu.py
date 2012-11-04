# $Id$ 
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""DXF tools

This module contains a set of tools for handling models read from DXF files.
The models currently handle only the ARC, LINE and POLYLINE items from the
DXF file.

These tools contain some extra functionality above the dxf plugin, which merely
handles import and export of the dxf model. The tools are mostly interactive.

While they are intende mostly to be imported in other applications, a plugin
menu is provided for standard tasks.
"""
from __future__ import print_function

from gui.draw import *
from plugins import dxf
from plugins import geometry_menu
from plugins.curve import *
from gui import menu
from connectivity import Connectivity

_name_ = 'dxf_menu'
    
def importDxf(convert=False,keep=False):
    """Import a DXF file.

    The user is asked for the name of a .DXF file. Depending on the
    parameters, the following processing is done:

    =======     =====     ================================================
    convert     keep      actions
    =======     =====     ================================================
    False       False     import DXF entities to pyFormex (default)
    False       True      import DXF and save intermediate .dxftext format
    True        any       convert .dxf to .dxftext only
    =======     =====     ================================================

    If convert == False, this function returns the list imported DXF entities.
    """
    fn = askFilename(filter=utils.fileDescription('dxf'))
    if not fn:
        return

    pf.GUI.setBusy()
    text = dxf.readDXF(fn)
    pf.GUI.setBusy(False)
    if text:
        if convert or keep:
            f = file(utils.changeExt(fn,'.dxftext'),'w')
            f.write(text)
            f.close()
        if not convert:
            return importDxfText(text)


def importSaveDxf():
    """Import a DXF file and save the intermediate .dxftext."""
    importDxf(keep=True)

    
def convertDxf():
    """Read a DXF file and convert to dxftext."""
    importDxf(convert=True)

    
def importDxfText(text=None):
    """Import a dxftext script or file.

    A dxftext script is a script containing only function calls that
    generate dxf entities. See :func:`dxf.convertDXF`.

    - Without parameter, the name of a .dxftext file is asked and the
      script is read from that file.
    - If `text` is a single line string, it is used as the filename of the
      script.
    - If `text` contains at least one newline character, it is interpreted
      as the dxf script text.
    """
    import types
    if text is None:
        fn = askFilename(filter=utils.fileDescription('dxftext'))
        if not fn:
            return
        text = open(fn).read()
    elif '\n' not in text:
        text = open(text).read()
        
    pf.GUI.setBusy()
    parts = dxf.convertDXF(text)
    print("Imported %s entities" % len(parts))
    coll = dxf.collectByType(parts)
    parts = [ p for p in parts if type(p) is not types.FunctionType ]
    print("Kept %s entities of type Arc, Line, PolyLine" % len(parts))
    pf.GUI.setBusy(False)
    export({'_dxf_import_':parts,'_dxf_sel_':parts})
    wireframe()
    drawDxf(zoom=True)
    return parts
   

def pickParts(filter=None):
    """Interactively pick a list of parts"""
    parts = named('_dxf_sel_')
    if not parts:
        return
    selection = pick('actor',filter=filter)
    #print selection
    try:
        selection = selection[-1]
        export ({'_last_selection_':selection})
        print("Selected entities: %s" % selection)
        return selection
    except:
        return None
    

def pickSelection():
    """Replace the parts by the picked selection"""
    import olist
    parts = named('_dxf_sel_')
    if not parts:
        return
    while True:
        drawDxf()
        selection = pickParts()
        if selection is None:
            break
        parts = olist.select(parts,selection)
        export({'_dxf_sel_':parts})
    drawDxf(zoom=True)


def printSelection():
    """Print the picked selection"""
    import olist
    parts = named('_dxf_sel_')
    if not parts:
        return
    drawParts()
    selection = pickParts()
    if selection is None:
        return
    selection = [ s for s in selection if s < len(parts) ]
    print("Selected entities: %s" % selection)
    for p in olist.select(parts,selection):
        print("PART %s: %s" % (p.prop,p))
        if hasattr(p,'endp'):
            print("  End Points: %s" % p.endp)
        if hasattr(p,'freep'):
            print("  Free Points: %s" % p.freep)
        if hasattr(p,'endd'):
            print("  End Directions: %s" % p.endd)


def renumberDxf():
    """Renumber the parts.

    Part numbers are stored in the prop attribute.
    """
    parts = named('_dxf_sel_')
    if parts:
        for i,p in enumerate(parts):
            p.setProp(i)
        drawCell()


def exportDxfText():
    """Export the parts as a .dxftext file"""
    parts = named('_dxf_sel_')
    if parts:
        types = utils.fileDescription(['dxftext'])
        cur = pf.cfg['workdir']
        fn = askFilename(cur=cur,filter=types,exist=False)
        if fn:
            dxf.exportDxfText(fn,parts)
            print("Wrote .dxftext file %s" % fn)


def exportDxf():
    """Export the parts as a .dxf file.

    Currently, pyFormex exports only Lines in .dxf format.
    Thus all parts will be approximated by Lines first.
    """
    parts = named('_dxf_sel_')
    if parts:
        utils.warn("warn_dxf_export")
        convertToFormex()
        exportLines()


def exportLines():
    """Export the lines as a .dxf file."""
    F = named("_lines_")
    if F:
        types = utils.fileDescription(['dxf'])
        cur = pf.cfg['workdir']
        fn = askFilename(cur=cur,filter=types,exist=False)
        if fn:
            dxf.exportDxf(fn,F)
            print("Wrote .dxf file %s" % fn)


def editArc(p):
    a = p.getAngles()
    res = askItems([
        _I('part_number',p.prop,readonly=True),
        _I('center',p._center,itemtype='point'),
        _I('radius',p.radius),
        _I('start_angle',a[0]),
        _I('end_angle',a[1]),
        ])
    if res:
        return Arc(center=res['center'],radius=res['radius'],angles=(res['start_angle'],res['end_angle'])).setProp(p.prop)


def editLine(p):
    x0,x1 = p.coords
    res = askItems([
        _I('part_number',p.prop,readonly=True),
        _I('point 0',x0,itemtype='point'),
        _I('point 1',x1,itemtype='point'),
        ])
    if res:
        return Line([x0,x1]).setProp(p.prop)
        

def editParts():
    """Edit the picked selection"""
    parts = named('_dxf_sel_')
    if not parts:
        return
    drawParts()
    selection = pickParts()
    if selection is None:
        return
    for i in selection:
        p = parts[i]
        try:
            newp = globals()['edit'+type(p).__name__](p)
        except:
            newp = None
        if newp:
            print("Replacing part %s" % p.prop)
            parts[i] = newp
                

def splitArcs():
    """Split the picked Arcs"""
    parts = named('_dxf_sel_')
    if not parts:
        return
    while True:
        drawParts(zoom=False,showpoints=True,shownumbers=True)
        selection = pickParts(filter='single')
        if selection is None:
            break
        i = selection[0]
        p = parts[i]
        a = p.getAngles()
        if type(p).__name__ == 'Arc':
            res = askItems([
                _I('part_number',p.prop,readonly=True),
                _I('center',p._center,itemtype='point',readonly=True),
                _I('radius',p.radius,readonly=True),
                _I('start_angle',a[0],readonly=True),
                _I('end_angle',a[1],readonly=True),
                _I('split_angle',(a[0]+a[1])/2)])      
            if res:
                asp = res['split_angle']
                newp = [ Arc(center=p._center,radius=p.radius,angles=ap).setProp(p.prop) for ap in [ (a[0],asp), (asp,a[1]) ] ]
                print("NEW: %s" % len(newp))
                parts[i:i+1] = newp
        print("# of parts: %s" % len(parts))
    drawCell()
                

def splitLines():
    """Split the picked Lines"""
    parts = named('_dxf_sel_')
    if not parts:
        return
    while True:
        drawParts(zoom=False,showpoints=True,shownumbers=True)
        selection = pickParts(filter='single')
        if selection is None:
            break
        i = selection[0]
        p = parts[i]
        x0,x1 = p.coords
        if type(p).__name__ == 'Line':
            res = askItems([
                _I('part_number',p.prop,readonly=True),
                _I('point 0',x0,itemtype='point'),
                _I('point 1',x1,itemtype='point'),
                _I('split parameter value',0.5)])      
            if res:
                asp = res['split parameter value']
                xm = (1.0-asp) * x0 + asp * x1
                newp = [ Line([x0,xm]).setProp(p.prop), Line([xm,x1]).setProp(p.prop) ]
                print("NEW: %s" % len(newp))
                parts[i:i+1] = newp
        print("# of parts: %s" % len(parts))
    drawCell()


def endPoints(parts):
    """Find the end points of all parts"""
    ep = Coords.concatenate([ p.coords[[0,-1]] for p in parts ])
    endpoints, ind = ep.fuse()
    ind = Connectivity(ind.reshape(-1,2))
    return endpoints,ind


def convertToFormex():
    """Convert all dxf parts to a plex-2 Formex

    This uses the :func:`dxf.toLines` function to transform all Lines, Arcs
    and PolyLines to a plex-2 Formex. 
    The parameters chordal and arcdiv used to set the precision of the
    Arc approximation are asked from the user..
    """
    parts = named('_dxf_sel_')
    if not parts:
        return
    res = askItems(
        [  _I('method',choices=['chordal error','fixed number'],tooltip="What method should be used to approximate the Arcs by straight line segments."),
           _I('chordal',0.01),
           _I('ndiv',8),
           ],
        enablers = [
            ('method','chordal error','chordal'),
            ('method','fixed number','ndiv'),
            ]
        )
    if res:
        chordal = res['chordal']
        if res['method'][0] == 'c':
            ndiv = None
        else:
            ndiv = res['ndiv']

        coll = dxf.collectByType(parts)
        lines = dxf.toLines(chordal=chordal,arcdiv=ndiv)
        print("Number of lines: %s" % _lines.nelems())
        export({'_dxf_lines_':lines})

        clear()
        draw(lines)

 
##########################################################################
######## Drawing #########


_show_colors = False
_show_props = True
_show_dirs = True
_show_endpoints = True
_show_freepoints = True
_show_bif = True
    



def toggleProps():
    global _show_props
    _show_props = not _show_props
def toggleColors():
    global _show_colors
    _show_colors = not _show_colors
def toggleEndpoints():
    global _show_props
    _show_props = not _show_props
def toggleBif():
    global _show_props
    _show_props = not _show_props



def drawParts(zoom=True,showpoints=None,shownumbers=None,showbif=None):
    """Draw parts"""
    parts = named('_dxf_sel_')
    if not parts:
        return
    if showpoints is None:
        showpoints = _show_endpoints
    clear()
    if _show_colors:
        draw(parts)
    else:
        draw(parts,color=black)
    if zoom:
        zoomAll()
    if shownumbers:
        nrs = array([p.prop for p in parts])
        X = Coords.concatenate([p.pointsAt([0.5])[0] for p in parts])
        drawMarks(X,nrs,color=blue)#,color=nrs)
    if showpoints:
        endpoints,ind = endPoints(parts)
        draw(endpoints)
        drawNumbers(endpoints)
    if showbif and 'bifurc' in pf.PF:
        bif = named('bifurc')
        print(bif)
        for b in bif:
            for i in b:
                p = parts[i]
                draw(p,color=red,linewidth=3)

           
def drawDxf():
    """Draw the imported dxf model"""
    dxf_parts = named('_dxf_import_')
    if dxf_parts:
        draw(dxf_parts,color='black',nolight=True)
 
def drawDxf(zoom=True):
    drawParts(zoom=zoom,showpoints=False,shownumbers=False,)


def drawDxf(zoom=True):
    drawParts(zoom=zoom,showpoints=False,shownumbers=False,showbif=False)


def drawCell(zoom=True,):
    drawParts(zoom=zoom,showpoints=True,shownumbers=True,showbif=True)


def drawDirs():
    x = Formex([p.pointsAt([0.5])[0] for p in parts])
    v = Formex([p.directionsAt([0.5])[0] for p in parts])
    drawVectors(x,v,color=red)
            

##########################################################################
######### Create a menu with interactive tasks #############
        
 
_menu = 'Dxf'

def create_menu():
    """Create the DXF menu."""
    MenuData = [
        ("&Read DXF file",importDxf),
        ("&Read DXFTEXT file",importDxfText),
        ("&Convert DXF to DXFTEXT",convertDxf,dict(tooltip="Parse a .dxf file and output a .dxftext script.")),
        ("&Read DXF without saving the DXFTEXT",importDxf),
        ("---",None),
        ("&Pick DXF selection",pickSelection),
        ("&Renumber DXF entities",renumberDxf),
        ("&Edit DXF entities",editParts),
        ("&Split Arcs",splitArcs),
        ("&Split Lines",splitLines),
        ("---",None),
        ("&Write DXFTEXT file",exportDxfText),
        ("&Convert to Formex",convertToFormex),
        ("&Write DXF file",exportDxf),
        #("&Clip by property",clipByProp),
        #("&Partition by connection",partition),
        ("---",None),
        ("&Print DXF selection",printSelection),
        ("&Draw DXF entities",drawDxf),
        ("&Toggle drawing property numbers",toggleProps),
        ("&Toggle drawing property colors",toggleColors),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item('Dxf'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = pf.GUI.menu.item('Dxf')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script/app is executed ###############

def run():
    _init_()
    reload_menu()

if __name__ == 'draw':
    run()
    
# End

