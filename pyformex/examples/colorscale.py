#!/usr/bin/env pyformex
# $Id$

"""Colorscale

level = 'advanced'
topics = ['FEA']
techniques = ['dialog', 'colors']

"""

from formex import *
from plugins import formex_menu

def drawcolorscale():
	res = askItems([('Title','Von mises stress [MPa]'),('Scale',None,'radio', ['Linear','Logarithmic']),('Color scale',None,'select', ['Rainbow','Inverse rainbow','Blue to red','Red to blue','Black to white','White to black']),('Continuous',False),('Number of intervals',12),('Minimum value',0.0),('Maximum value',12.0),('Precision',2),('Show every',3),('Linewidth',2)])
	if not res: return
	n = res['Number of intervals']
	if n<2 or n>24:
		warning('Selected number of intervals (%s) is not supported by Abaqus' %n)
		if n<2: return
	mi = res['Minimum value']
	ma = res['Maximum value']
	if not mi<ma:
		warning('The minimum value should be smaller than the maximum value')
		return
	p = res['Precision']
	s = res['Show every']
	lin = res['Scale']=='Linear'
	if not lin and mi==0: lin=True
	col = res['Color scale']
	txt = str(res['Title'])
	con = res['Continuous']
	if con: intercol = arange(0.,1.+1./(n+1),1./n)
	else: intercol = arange(0.,1.+1./n,1./(n-1))
	if lin: interval = arange(mi,ma+(ma-mi)/(n+1),(ma-mi)/n)
	else: interval = [10.**((1-i)*log10(mi)+i*log10(ma)) for i in arange(0.,1.+1./(n+1),1./n)]
	if col=='Rainbow': color=[[4*i-2,min(4*i,4-4*i),2-4*i] for i in intercol]
	elif col=='Inverse rainbow': color=[[2-4*i,min(4*i,4-4*i),4*i-2] for i in intercol]
	elif col=='Blue to red': color=[[i**2,23./255.,(1-i)**2] for i in intercol]
	elif col=='Red to blue': color=[[(1-i)**2,23./255.,i**2] for i in intercol]
	elif col=='Black to white': color=[[38./255.+i*(1.-2.*38./255.),38./255.+i*(1.-2.*38./255.),38./255.+i*(1.-2.*38./255.)] for i in intercol]
	elif col=='White to black': color=[[1-38./255.-i*(1.-2.*38./255.),1-38./255.-i*(1.-2.*38./255.),1-38./255.-i*(1.-2.*38./255.)] for i in intercol]
	F = Formex([[[0,0,0],[2,0,0],[2,1,0],[0,1,0]]]).replic(n,1,dir=1)
	if con: color = resize([[color[i],color[i],color[i+1],color[i+1]] for i in arange(len(color)-1)],F.shape())
	G = Formex([[[0,0,0],[0,n,0]]])
	G += G.translate([2,0,0]) + Formex([[[0,0,0],[2.5,0,0]]]).replic(n+1,1,dir=1) + Formex([[[0,0,0],[3.0,0,0]]]).replic((n+s)/s,s,dir=1) + Formex([[[0,n,0],[3.0,n,0]]])
	draw(G,linewidth=res['Linewidth'])
	draw(F,color=color)
	smooth()
	lights(False)
	for i in range(n+1): 
		if i%s==0 or i==n: drawText3D([3.1,i-0.1,0],str(round(interval[i],p)),size=30.0)
	drawText3D([0,i+1.0,0],txt,size=20.0)
	zoomAll()

if __name__ == 'draw':
	clear()
	view('front')
	bgcolor('white')
	drawcolorscale()
