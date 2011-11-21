/* $Id$ */
//
//  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
//  pyFormex is a tool for generating, manipulating and transforming 3D
//  geometrical models by sequences of mathematical operations.
//  Home page: http://pyformex.org
//  Project page:  https://savannah.nongnu.org/projects/pyformex/
//  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
//  Distributed under the GNU General Public License version 3 or later.
//
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see http://www.gnu.org/licenses/.
//


/**************************************************************************
  dxfparser.cc      DXF parser
 
  dxfparser uses the dxflib from QCAD to parse an AutoCAD .DXF file,
  and exports it as a script with function calling syntax. 
 
***************************************************************************/

#include "dxfparser.h"

#include <stdio.h>

const char* _version_ = "dxfparser 0.1";
const char* _copyright_ = "Copyright (C) 2011 Benedict Verhegghe";
const char* LineFmt = "Line(%f,%f,%f,%f,%f,%f)\n";
const char* ArcFmt = "Arc(%f,%f,%f,%f,%f,%f)\n";

void MyDxfFilter::addLine(const DL_LineData& d) {
  printf(LineFmt,d.x1,d.y1,d.z1,d.x2,d.y2,d.z2);
}

void MyDxfFilter::addArc(const DL_ArcData& d) {
  printf(ArcFmt,d.cx,d.cy,d.cz,d.radius,d.angle1,d.angle2);
}


int main(int argc, char* argv[])
{
  MyDxfFilter f;
  DL_Dxf dxf;
  for(int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      // Process switch
      if (!strcmp(argv[i],"-v")) {
	printf("%s\n",_version_);
	return 0;
      }
      printf("Unknown switch '%s'\n",argv[i]);
      return 1;
    }
    printf("# Converting %s\n",argv[i]);
    if (!dxf.in(argv[i], &f)) {
      printf(" !! file could not be opened.\n");
      return 1;
    }
  }
  return 0;
}

// End
