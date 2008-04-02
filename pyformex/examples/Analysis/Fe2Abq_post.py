#!/usr/bin/env pyformex
# Created by postabq 0.1 (C) 2008 Benedict Verhegghe
from plugins.postabq import *
Initialize()
Abqver('6.7-3')
Date('24-Mar-2008','14:33:11')
Size(nelems=44,nnodes=44,length=1.150623)
Element(1,'CPE3',[11,16,17,])
Element(2,'CPE3',[17,12,11,])
Element(3,'CPE3',[16,19,20,])
Element(4,'CPE3',[20,17,16,])
Element(5,'CPE3',[19,22,23,])
Element(6,'CPE3',[23,20,19,])
Element(7,'CPE3',[22,25,26,])
Element(8,'CPE3',[26,23,22,])
Element(9,'CPE3',[12,17,18,])
Element(10,'CPE3',[18,13,12,])
Element(11,'CPE3',[17,20,21,])
Element(12,'CPE3',[21,18,17,])
Element(13,'CPE3',[20,23,24,])
Element(14,'CPE3',[24,21,20,])
Element(15,'CPE3',[23,26,27,])
Element(16,'CPE3',[27,24,23,])
Element(17,'CPE4',[25,30,31,26,])
Element(18,'CPE4',[35,40,41,36,])
Element(19,'CPE4',[31,36,37,32,])
Element(20,'CPE4',[27,32,33,28,])
Element(21,'CPE4',[37,42,43,38,])
Element(22,'CPE4',[33,38,39,34,])
Element(23,'CPE4',[30,35,36,31,])
Element(24,'CPE4',[26,31,32,27,])
Element(25,'CPE4',[36,41,42,37,])
Element(26,'CPE4',[32,37,38,33,])
Element(27,'CPE4',[28,33,34,29,])
Element(28,'CPE4',[38,43,44,39,])
Element(29,'CPE3',[11,12,7,])
Element(30,'CPE3',[7,6,11,])
Element(31,'CPE3',[12,13,8,])
Element(32,'CPE3',[8,7,12,])
Element(33,'CPE3',[13,14,9,])
Element(34,'CPE3',[9,8,13,])
Element(35,'CPE3',[14,15,10,])
Element(36,'CPE3',[10,9,14,])
Element(37,'CPE3',[6,7,2,])
Element(38,'CPE3',[2,1,6,])
Element(39,'CPE3',[7,8,3,])
Element(40,'CPE3',[3,2,7,])
Element(41,'CPE3',[8,9,4,])
Element(42,'CPE3',[4,3,8,])
Element(43,'CPE3',[9,10,5,])
Element(44,'CPE3',[5,4,9,])
Node(1,[-2.000000e+00,1.224647e-16,])
Node(2,[-2.000000e+00,1.000000e+00,])
Node(3,[-2.000000e+00,2.000000e+00,])
Node(4,[-2.000000e+00,3.000000e+00,])
Node(5,[-2.000000e+00,4.000000e+00,])
Node(6,[-1.000000e+00,6.123234e-17,])
Node(7,[-1.000000e+00,1.000000e+00,])
Node(8,[-1.000000e+00,2.000000e+00,])
Node(9,[-1.000000e+00,3.000000e+00,])
Node(10,[-1.000000e+00,4.000000e+00,])
Node(11,[0.000000e+00,0.000000e+00,])
Node(12,[6.123234e-17,1.000000e+00,])
Node(13,[0.000000e+00,2.000000e+00,])
Node(14,[1.836970e-16,3.000000e+00,])
Node(15,[2.449294e-16,4.000000e+00,])
Node(16,[1.000000e+00,0.000000e+00,])
Node(17,[1.000000e+00,1.000000e+00,])
Node(18,[1.000000e+00,2.000000e+00,])
Node(19,[2.000000e+00,0.000000e+00,])
Node(20,[2.000000e+00,1.000000e+00,])
Node(21,[2.000000e+00,2.000000e+00,])
Node(22,[3.000000e+00,0.000000e+00,])
Node(23,[3.000000e+00,1.000000e+00,])
Node(24,[3.000000e+00,2.000000e+00,])
Node(25,[4.000000e+00,0.000000e+00,])
Node(26,[4.000000e+00,1.000000e+00,])
Node(27,[4.000000e+00,2.000000e+00,])
Node(28,[4.000000e+00,3.000000e+00,])
Node(29,[4.000000e+00,4.000000e+00,])
Node(30,[5.000000e+00,0.000000e+00,])
Node(31,[5.000000e+00,1.000000e+00,])
Node(32,[5.000000e+00,2.000000e+00,])
Node(33,[5.000000e+00,3.000000e+00,])
Node(34,[5.000000e+00,4.000000e+00,])
Node(35,[6.000000e+00,0.000000e+00,])
Node(36,[6.000000e+00,1.000000e+00,])
Node(37,[6.000000e+00,2.000000e+00,])
Node(38,[6.000000e+00,3.000000e+00,])
Node(39,[6.000000e+00,4.000000e+00,])
Node(40,[7.000000e+00,0.000000e+00,])
Node(41,[7.000000e+00,1.000000e+00,])
Node(42,[7.000000e+00,2.000000e+00,])
Node(43,[7.000000e+00,3.000000e+00,])
Node(44,[7.000000e+00,4.000000e+00,])
Elemset('ESET_1_1',[17,18,19,20,21,22,])
Elemset('ESET_1_2',[23,24,25,26,27,28,])
Elemset('ESET_0_3',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,])
Elemset('ESET_1',[17,18,19,20,21,22,])
Elemset('ESET_2',[23,24,25,26,27,28,])
Elemset('ESET_3',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,])
Elemset('ESET_5',[29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,])
Elemset('EALL',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,])
Elemset('ESET_2_5',[29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,])
Nodeset('NALL',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,])
Nodeset('NSET_1',[40,41,42,43,44,])
Nodeset('NSET_6',[1,2,3,4,5,])
Label(tag='1',value='ANTIALIASING')
Dofs([1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,])
Heading('Model: Fe2Abq     Date: 2008-03-24      Created by pyFormex')
EndIncrement()
Increment(step=1,inc=1,tottime=1.000000e+00,steptime=1.000000e+00,timeinc=1.000000e+00,type=1,heading='',maxcreep=0.000000e+00,solamp=0.000000e+00,linpert=0,loadfactor=0.000000e+00,frequency=0.000000e+00,)
OutputRequest(flag=6303040,set='(null)',eltyp='CPE3',)
ElemHeader(loc='gp',ie=1,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-1.515541e+03,-9.105926e+02,-7.278401e+02,7.318950e+02,])
ElemHeader(loc='gp',ie=2,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.506688e+03,4.642079e+02,8.912686e+02,-3.475108e+02,])
ElemHeader(loc='gp',ie=3,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-1.691145e+03,-9.269824e+02,-7.854383e+02,7.104969e+02,])
ElemHeader(loc='gp',ie=4,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.783820e+03,9.319907e+02,1.114743e+03,-5.562907e+02,])
ElemHeader(loc='gp',ie=5,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-1.776945e+03,-5.772319e+02,-7.062530e+02,6.192921e+02,])
ElemHeader(loc='gp',ie=6,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.847584e+03,1.018187e+03,1.159731e+03,-6.246975e+02,])
ElemHeader(loc='gp',ie=7,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-1.574922e+03,1.095274e+02,-4.396185e+02,1.100599e+02,])
ElemHeader(loc='gp',ie=8,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.105013e+03,1.086464e+03,9.574430e+02,-8.213145e+02,])
ElemHeader(loc='gp',ie=9,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.889756e+03,-9.753002e+02,2.743366e+02,6.334661e+02,])
ElemHeader(loc='gp',ie=10,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[7.119098e+03,2.051939e+03,2.751311e+03,-1.017850e+03,])
ElemHeader(loc='gp',ie=11,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.952444e+03,-1.007887e+03,2.833671e+02,6.434761e+02,])
ElemHeader(loc='gp',ie=12,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.954881e+03,1.195468e+03,2.445105e+03,-7.976824e+02,])
ElemHeader(loc='gp',ie=13,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.020116e+03,-9.125714e+02,3.322634e+02,6.945184e+02,])
ElemHeader(loc='gp',ie=14,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.909245e+03,1.116456e+03,2.407710e+03,-6.891129e+02,])
ElemHeader(loc='gp',ie=15,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.078359e+03,1.024271e+03,9.307891e+02,9.234666e+02,])
ElemHeader(loc='gp',ie=16,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[7.391551e+03,1.389472e+03,2.634307e+03,-2.122121e+02,])
ElemHeader(loc='gp',ie=29,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.069224e+03,-1.518478e+02,2.752130e+02,9.653652e+02,])
ElemHeader(loc='gp',ie=30,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-2.766693e+02,-4.050095e+02,-2.045036e+02,6.455541e+02,])
ElemHeader(loc='gp',ie=31,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.933112e+03,-1.706261e+02,5.287458e+02,5.082996e+02,])
ElemHeader(loc='gp',ie=32,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.577791e+02,-4.118870e+02,1.637676e+02,2.649808e+02,])
ElemHeader(loc='gp',ie=33,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.935916e+02,1.476735e+03,6.510980e+02,-9.155811e+02,])
ElemHeader(loc='gp',ie=34,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.067639e+03,1.432701e+02,6.632728e+02,-9.074646e+02,])
ElemHeader(loc='gp',ie=35,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.220059e+02,2.220059e+02,1.332035e+02,-2.220059e+02,])
ElemHeader(loc='gp',ie=36,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-1.632993e-02,-1.416832e+02,-4.250986e+01,-3.391482e+02,])
ElemHeader(loc='gp',ie=37,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.064110e+03,1.696100e+02,3.701159e+02,5.542938e+02,])
ElemHeader(loc='gp',ie=38,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.231786e+02,-4.101547e+02,1.539072e+02,4.101547e+02,])
ElemHeader(loc='gp',ie=39,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.166815e+03,-3.223001e+02,2.533545e+02,4.740872e+02,])
ElemHeader(loc='gp',ie=40,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[5.021140e+02,-1.141713e+03,-1.918797e+02,1.772645e+02,])
ElemHeader(loc='gp',ie=41,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.264409e+02,-4.743864e+02,4.561636e+01,-3.734396e+02,])
ElemHeader(loc='gp',ie=42,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[8.139642e+02,-1.145619e+03,-9.949641e+01,-4.701814e+02,])
ElemHeader(loc='gp',ie=43,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[8.746509e+02,2.331741e+02,3.323475e+02,-3.134968e+02,])
ElemHeader(loc='gp',ie=44,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.953936e+02,-3.134968e+02,1.145690e+02,-4.586825e+02,])
OutputRequest(flag=6303040,set='(null)',eltyp='CPE4',)
ElemHeader(loc='gp',ie=17,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-5.851172e+02,5.450476e+02,-1.202088e+01,-4.637472e+02,])
ElemHeader(loc='gp',ie=17,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-3.933884e+02,3.533188e+02,-1.202088e+01,2.453727e+01,])
ElemHeader(loc='gp',ie=17,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-9.683273e+01,5.676311e+01,-1.202088e+01,-6.554761e+02,])
ElemHeader(loc='gp',ie=17,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.489612e+01,-1.349657e+02,-1.202088e+01,-1.671916e+02,])
ElemHeader(loc='gp',ie=18,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.171681e+02,5.539140e+01,2.017679e+02,-2.303671e+02,])
ElemHeader(loc='gp',ie=18,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[5.549622e+02,1.175973e+02,2.017679e+02,-2.454265e+02,])
ElemHeader(loc='gp',ie=18,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.021088e+02,7.045072e+01,2.017679e+02,-1.681612e+02,])
ElemHeader(loc='gp',ie=18,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[5.399029e+02,1.326566e+02,2.017679e+02,-1.832205e+02,])
ElemHeader(loc='gp',ie=19,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.047364e+03,-2.854990e+02,2.285596e+02,-2.447077e+02,])
ElemHeader(loc='gp',ie=19,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.488758e+02,-1.870104e+02,2.285596e+02,-1.856286e+02,])
ElemHeader(loc='gp',ie=19,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.106443e+03,-3.445781e+02,2.285596e+02,-1.462191e+02,])
ElemHeader(loc='gp',ie=19,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.007955e+03,-2.460895e+02,2.285596e+02,-8.713996e+01,])
ElemHeader(loc='gp',ie=20,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.278131e+03,2.321490e+02,4.530840e+02,9.686053e+02,])
ElemHeader(loc='gp',ie=20,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.496483e+03,1.379678e+01,4.530840e+02,4.118732e+02,])
ElemHeader(loc='gp',ie=20,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[7.213991e+02,7.888810e+02,4.530840e+02,7.502531e+02,])
ElemHeader(loc='gp',ie=20,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.397513e+02,5.705289e+02,4.530840e+02,1.935210e+02,])
ElemHeader(loc='gp',ie=21,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[7.614957e+02,-2.607980e+00,2.276663e+02,9.044936e+01,])
ElemHeader(loc='gp',ie=21,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.682322e+02,9.065556e+01,2.276663e+02,4.623869e+01,])
ElemHeader(loc='gp',ie=21,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[7.172850e+02,4.160270e+01,2.276663e+02,1.837129e+02,])
ElemHeader(loc='gp',ie=21,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.240215e+02,1.348662e+02,2.276663e+02,1.395022e+02,])
ElemHeader(loc='gp',ie=22,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[3.030965e+02,3.718972e+01,1.020859e+02,3.657612e+02,])
ElemHeader(loc='gp',ie=22,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[4.319514e+02,-9.166525e+01,1.020859e+02,1.581770e+02,])
ElemHeader(loc='gp',ie=22,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.551225e+01,2.447740e+02,1.020859e+02,2.369062e+02,])
ElemHeader(loc='gp',ie=22,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[2.243672e+02,1.159190e+02,1.020859e+02,2.932200e+01,])
ElemHeader(loc='gp',ie=23,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.161203e+01,2.401914e+01,3.468935e+01,-4.215771e+02,])
ElemHeader(loc='gp',ie=23,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.206139e+02,-4.982715e+00,3.468935e+01,-1.414095e+02,])
ElemHeader(loc='gp',ie=23,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[3.717797e+02,-2.561485e+02,3.468935e+01,-4.505790e+02,])
ElemHeader(loc='gp',ie=23,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[4.007816e+02,-2.851504e+02,3.468935e+01,-1.704114e+02,])
ElemHeader(loc='gp',ie=24,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.484504e+03,5.921590e+02,6.229990e+02,-2.747158e+02,])
ElemHeader(loc='gp',ie=24,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.703442e+03,3.732209e+02,6.229990e+02,1.612486e+01,])
ElemHeader(loc='gp',ie=24,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.775345e+03,3.013183e+02,6.229990e+02,-4.936538e+02,])
ElemHeader(loc='gp',ie=24,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.994283e+03,8.238030e+01,6.229990e+02,-2.028132e+02,])
ElemHeader(loc='gp',ie=25,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.785608e+02,3.217046e+01,2.132194e+02,-2.070656e+02,])
ElemHeader(loc='gp',ie=25,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[5.850974e+02,1.256338e+02,2.132194e+02,-1.519335e+02,])
ElemHeader(loc='gp',ie=25,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[7.336929e+02,-2.296167e+01,2.132194e+02,-1.136022e+02,])
ElemHeader(loc='gp',ie=25,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.402296e+02,7.050169e+01,2.132194e+02,-5.847010e+01,])
ElemHeader(loc='gp',ie=26,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.032719e+03,-1.957141e+02,2.511015e+02,3.038337e+02,])
ElemHeader(loc='gp',ie=26,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[1.001869e+03,-1.648641e+02,2.511015e+02,1.940692e+02,])
ElemHeader(loc='gp',ie=26,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[9.229545e+02,-8.594959e+01,2.511015e+02,3.346838e+02,])
ElemHeader(loc='gp',ie=26,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[8.921044e+02,-5.509951e+01,2.511015e+02,2.249192e+02,])
ElemHeader(loc='gp',ie=27,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-1.259742e+02,4.604632e+01,-2.397838e+01,-1.363994e+02,])
ElemHeader(loc='gp',ie=27,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-2.127945e+02,1.328666e+02,-2.397838e+01,-4.079369e+00,])
ElemHeader(loc='gp',ie=27,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.345839e+00,-8.627376e+01,-2.397838e+01,-4.957921e+01,])
ElemHeader(loc='gp',ie=27,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[-8.047440e+01,5.464817e-01,-2.397838e+01,8.274088e+01,])
ElemHeader(loc='gp',ie=28,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.080677e+02,2.757987e+01,1.906943e+02,1.915451e+02,])
ElemHeader(loc='gp',ie=28,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[5.362699e+02,9.937766e+01,1.906943e+02,1.858289e+02,])
ElemHeader(loc='gp',ie=28,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[6.023516e+02,3.329603e+01,1.906943e+02,2.633429e+02,])
ElemHeader(loc='gp',ie=28,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
ElemOutput('S',[5.305538e+02,1.050938e+02,1.906943e+02,2.576267e+02,])
OutputRequest(flag=6303040,set='(null)',eltyp='CPE4',)
ElemHeader(loc='gp',ie=17,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=17,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=17,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=17,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=18,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=18,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=18,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=18,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=19,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=19,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=19,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=19,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=20,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=20,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=20,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=20,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=21,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=21,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=21,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=21,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=22,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=22,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=22,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
ElemHeader(loc='gp',ie=22,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(401)
OutputRequest(flag=6303040,set='(null)',eltyp='CPE4',)
ElemHeader(loc='gp',ie=17,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=17,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=17,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=17,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=18,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=18,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=18,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=18,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=19,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=19,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=19,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=19,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=20,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=20,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=20,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=20,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=21,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=21,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=21,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=21,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=22,gp=1,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=22,gp=2,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=22,gp=3,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
ElemHeader(loc='gp',ie=22,gp=4,sp=0,ndi=3,nshr=1,nsfc=0,)
Unknown(12)
OutputRequest(flag=6303040,set='(null)')
NodeOutput('U',1,[2.607553e-02,-3.740437e-01,])
NodeOutput('U',2,[-5.820498e-02,-3.775861e-01,])
NodeOutput('U',3,[-1.476939e-01,-3.835512e-01,])
NodeOutput('U',4,[-2.476649e-01,-3.901211e-01,])
NodeOutput('U',5,[-3.507956e-01,-3.928094e-01,])
NodeOutput('U',6,[3.090671e-02,-2.846115e-01,])
NodeOutput('U',7,[-5.384656e-02,-2.858707e-01,])
NodeOutput('U',8,[-1.419572e-01,-2.894859e-01,])
NodeOutput('U',9,[-2.440172e-01,-2.927516e-01,])
NodeOutput('U',10,[-3.473898e-01,-2.933744e-01,])
NodeOutput('U',11,[3.045349e-02,-1.917498e-01,])
NodeOutput('U',12,[-4.886002e-02,-1.944318e-01,])
NodeOutput('U',13,[-1.331375e-01,-1.988240e-01,])
NodeOutput('U',14,[-2.437503e-01,-1.936388e-01,])
NodeOutput('U',15,[-3.468321e-01,-1.930811e-01,])
NodeOutput('U',16,[2.550658e-02,-1.183355e-01,])
NodeOutput('U',17,[-3.871487e-02,-1.194832e-01,])
NodeOutput('U',18,[-1.057069e-01,-1.273311e-01,])
NodeOutput('U',19,[1.981856e-02,-6.136002e-02,])
NodeOutput('U',20,[-2.823275e-02,-6.224895e-02,])
NodeOutput('U',21,[-7.738466e-02,-7.035827e-02,])
NodeOutput('U',22,[1.309441e-02,-2.285436e-02,])
NodeOutput('U',23,[-1.763271e-02,-2.204408e-02,])
NodeOutput('U',24,[-4.911416e-02,-2.986188e-02,])
NodeOutput('U',25,[5.964483e-03,-5.081730e-03,])
NodeOutput('U',26,[-1.042575e-02,-1.632988e-03,])
NodeOutput('U',27,[-1.923774e-02,-1.045900e-03,])
NodeOutput('U',28,[-3.776145e-03,1.004281e-03,])
NodeOutput('U',29,[-2.014400e-03,3.567786e-04,])
NodeOutput('U',30,[-8.107351e-04,4.120267e-03,])
NodeOutput('U',31,[-6.578227e-03,3.397904e-03,])
NodeOutput('U',32,[-9.062918e-03,-7.780544e-04,])
NodeOutput('U',33,[-5.713152e-03,-3.478174e-03,])
NodeOutput('U',34,[-1.072753e-03,-2.236883e-03,])
NodeOutput('U',35,[-2.529983e-03,3.437872e-03,])
NodeOutput('U',36,[-2.202364e-03,2.084568e-03,])
NodeOutput('U',37,[-3.401776e-03,5.125107e-05,])
NodeOutput('U',38,[-2.439963e-03,-1.977719e-03,])
NodeOutput('U',39,[-2.315606e-03,-3.539696e-03,])
NodeOutput('U',40,[-7.940238e-36,3.367325e-36,])
NodeOutput('U',41,[-1.123380e-35,3.149202e-36,])
NodeOutput('U',42,[-1.161766e-35,3.246974e-37,])
NodeOutput('U',43,[-1.130234e-35,-3.517472e-36,])
NodeOutput('U',44,[-7.905964e-36,-3.323752e-36,])
EndIncrement()

#print DB.nodes
#print DB.elems

Fn = Formex(DB.nodes)
draw(Fn)

## print DB.nodes.shape
## for i,elems in enumerate(DB.elems.itervalues()):
##     print elems.shape
##     nodes = DB.nodes[elems],i) 
## draw(Fe)

Fe = [ Formex(DB.nodes[elems],i+1) for i,elems in enumerate(DB.elems.itervalues()) ]
draw(Fe)
   
from plugins.postproc import *
from gui.colorscale import ColorScale,ColorLegend
import gui.decors
import gui.canvas

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
    if val is None:
        # only display deformed geometry
        val = 'blue'
    else:
        # create a colorscale and draw the colorlegend
        vmin,vmax = val.min(),val.max()
        if vmin*vmax < 0.0:
            vmid = 0.0
        else:
            vmid = 0.5*(vmin+vmax)
        CS = ColorScale([blue,green,red],vmin,vmax,vmid,1.,1.)
##         CS = ColorScale([green,None,magenta],0.,1.,None,0.5,None)
##         val = arange(11)/10.
        cval = array(map(CS.color,val))
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,10,20,30,200) 
        GD.canvas.addDecoration(CLA)

    # the supplied text
    if text:
        drawtext(text,150,30,'tr24')

    smoothwire()
    lights(False)
    # create the frames while displaying them
    dscale = array(dscale)
    frames = []   # a place to store the drawn frames
    for dsc in dscale.flat:

        dnodes = nodes + dsc * displ
        deformed = [ Formex(dnodes[el]) for el in elems ]

        # We store the changing parts of the display, so that we can
        # easily remove/redisplay them
        F = [ draw(df,color=cval[el],view='__last__',wait=None) for df,el in zip(deformed,elems) ]
        T = drawtext('Deformation scale = %s' % dsc,150,10,'tr18')

        # remove the last frame
        # This is a clever trick: we remove the old drawings only after
        # displaying new ones. This makes the animation a lot smoother
        # (though the code is less clear and compact).
        if len(frames) > 0:
            GD.canvas.removeActor(frames[-1][0])
            GD.canvas.removeDecoration(frames[-1][1])
        # add the latest frame to the stored list of frames
        frames.append((F,T))
        if sleeptime > 0.:
            sleep(sleeptime)

    # display the remaining cycles
    count -= 1
    FA,TA = frames[-1]
    #print frames
    #print count
    while count > 0:
        count -= 1

        for F,T in frames:
            #print count,F,T
            GD.canvas.addActor(F)
            GD.canvas.addDecoration(T)
            GD.canvas.removeActor(FA)
            GD.canvas.removeDecoration(TA)
            GD.canvas.display()
            GD.canvas.update()
            FA,TA = F,T
            if sleeptime > 0.:
                sleep(sleeptime)


def postCalpy():
    """Show results from the Calpy analysis."""
    try:
        nodes,tubes,nodeprops,tubeprops,botnodes,jobname = named('fe_model')
        print "OK"
        displ,frc = named('calpy_results')
        print "OK2"
    except:
        warning("I could not find the finite element model and/or the calpy results. Maybe you should try to first create them?")
        raise
        return
    
    # The frc array returns element forces and has shape
    #  (nelems,nforcevalues,nloadcases)
    # nforcevalues = 8 (Nx,Vy,Vz,Mx,My1,Mz1,My2,Mz2)
    # Describe the nforcevalues element results in frc.
    # For each result we give a short and a long description:
    frc_contents = [('Nx','Normal force'),
                    ('Vy','Shear force in local y-direction'),
                    ('Vz','Shear force in local z-direction'),
                    ('Mx','Torsional moment'),
                    ('My','Bending moment around local y-axis'),
                    ('Mz','Bending moment around local z-axis'),
                    ('None','No results'),
                    ]
    # split in two lists
    frc_keys = [ c[0] for c in frc_contents ]
    frc_desc = [ c[1] for c in frc_contents ]

    # Ask the user which results he wants
    res = askItems([('Type of result',None,'select',frc_desc),
                    ('Load case',0),
                    ('Autocalculate deformation scale',True),
                    ('Deformation scale',100.),
                    ('Show undeformed configuration',False),
                    ('Animate results',False),
                    ('Amplitude shape','linear','select',['linear','sine']),
                    ('Animation cycle','updown','select',['up','updown','revert']),
                    ('Number of cycles',5),
                    ('Number of frames',10),
                    ('Animation sleeptime',0.1),
                    ])
    if res:
        frcindex = frc_desc.index(res['Type of result'])
        loadcase = res['Load case']
        autoscale = res['Autocalculate deformation scale']
        dscale = res['Deformation scale']
        showref = res['Show undeformed configuration']
        animate = res['Animate results']
        shape = res['Amplitude shape']
        cycle = res['Animation cycle']
        count = res['Number of cycles']
        nframes = res['Number of frames']
        sleeptime = res['Animation sleeptime']

        dis = displ[:,0:3,loadcase]
        if autoscale:
            siz0 = Coords(nodes).sizes()
            siz1 = Coords(dis).sizes()
            print siz0
            print siz1
            dscale = niceNumber(1./(siz1/siz0).max())

        if animate:
            dscale = dscale * frameScale(nframes,cycle=cycle,shape=shape) 
        
##         # Get the scalar element result values from the frc array.
##         val = val1 = txt = None
##         if frcindex <= 5:
##             val = frc[:,frcindex,loadcase]
##             txt = frc_desc[frcindex]
##             if frcindex > 3:
##                 # bending moment values at second node
##                 val1 = frc[:,frcindex+2,loadcase]

        showResults(nodes,tubes,dis,txt,val,showref,dscale,count,sleeptime)


def renderDistanceFromPoint(elems,pt):
    """Show distance from origin rendered on the domain of triangles"""
    val = Fn.distanceFromPoint(pt)
    nodes = DB.nodes
    displ = zeros(nodes.shape)
    text = "Distance from point %s" % pt
    showResults(nodes,elems,displ,text,val,showref=False,dscale=100.,
                count=1,sleeptime=-1.)


def showGeom(elems):
    """Show wireframe geometry of element groups."""
    clear()
    wireframe()
    draw([Formex(DB.nodes[el]) for el in elems])
    zoomAll()
    
        
if __name__ == "draw":

    res = askItems([('Point',[3.,2.,0.]),
                    ('Element Group','All','select',['All','CPE3','CPE4']),
                    ])
    if res:
        point = res['Point']
        elgrp = res['Element Group']
        if elgrp == 'All':
            elems = DB.elems.values()
        else:
            elems = [ DB.elems[elgrp] ]
        showGeom(elems)
        renderDistanceFromPoint(elems,point)

# End
