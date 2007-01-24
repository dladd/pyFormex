#!/usr/bin/env awk
# Extract the nodal coordinates from a Gambit neutral file
#
BEGIN { 
  mode = 0; 
  if (nodes == "") nodes="nodes.out"; 
  if (elems == "") elems="elems.out"; 
}

/^[ ]*NODAL COORDINATES/ { mode = 1; next; }
/^[ ]*ELEMENTS/ { mode = 2; next; }
/^[ ]*ENDOF/ { mode = 0; next; }


{ if (mode == 1) print $2,$3,$4 >> nodes;
  else if (mode == 2) print $4,$5,$6 >> elems;
  else next; 
}
