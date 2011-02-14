#!/usr/bin/env pyformex

"""Exporting finite element models in code Aster mesh file format (.mail).

"""



def writeNodes(fil,nodes,type='COOR_2D'):
    """Write nodal coordinates.
    
    """
    fil.write('%s\n' % type)
    if type == 'COOR_2D':
        nodes = nodes[:,:2]
    nn = nodes.shape[1]
    fmt = 'N%d' + nn*' %14.6e' + '\n'
    for i,n in enumerate(nodes):
        fil.write(fmt % ((i,)+tuple(n)))
    fil.write('FINSF\n')
    fil.write('%\n')

def writeElems(fil,elems,type,name=None,eofs=0):
    """Write element group of given type.

    elems is the list with the element node numbers.
    type specifies the element type.
    eofs specifies the offset for the element numbers.
    """
    out = type
    if name is not None:
        out += ' nom = %s' % name
    out += '\n'
    fil.write(out)
    nn = elems.shape[1]
    fmt = 'M%d' + nn*' N%d' + '\n'
    for i,e in enumerate(elems):
        fil.write(fmt % ((i+eofs,)+tuple(e)))
    fil.write('FINSF\n')
    fil.write('%\n')

def writeSet(fil,type,name,set,ofs=0):
    """Write a named set of nodes or elements (type=NSET|ELSET)

    `set` is a list of node/element numbers,
    in which case the `ofs` value will be added to them.
    """
    if type == 'NSET':
        fil.write('GROUP_NO nom = %s\n' % name)
        cap = 'N'
    elif type == 'ELSET':
        fil.write('GROUP_MA nom = %s\n' % name)
        cap = 'M'
    else:
        raise ValueError,"Type should be NSET or ELSET"
        
    for i in set:
        fil.write('%s%d\n' % (cap,i))
    fil.write('FINSF\n')
    fil.write('%\n')


  
# End
