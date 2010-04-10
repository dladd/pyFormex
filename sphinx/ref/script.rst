.. $Id$  -*- rst -*-
.. pyformex reference manual --- script
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-script:

:mod:`script` --- Basic pyFormex script functions
=================================================

.. automodule:: script
   :synopsis: Basic pyFormex script functions



**Functions defined in the module script**

   .. autofunction:: Globals()
   .. autofunction:: export(dic)
   .. autofunction:: export2(names,values)
   .. autofunction:: forget(names)
   .. autofunction:: rename(oldnames,newnames)
   .. autofunction:: listAll(clas=None,like=None,filter=None,dic=None)
   .. autofunction:: named(name)
   .. autofunction:: getcfg(name)
   .. autofunction:: ask(question,choices=None,default='')
   .. autofunction:: ack(question)
   .. autofunction:: error(message)
   .. autofunction:: warning(message)
   .. autofunction:: showInfo(message)
   .. autofunction:: system(cmdline,result='output')
   .. autofunction:: playScript(scr,name=None,filename=None,argv=[],pye=False)
   .. autofunction:: force_finish()
   .. autofunction:: step_script(s,glob,paus=True)
   .. autofunction:: breakpt(msg=None)
   .. autofunction:: enableBreak(mode=True)
   .. autofunction:: stopatbreakpt()
   .. autofunction:: playFile(fn,argv=[])
   .. autofunction:: play(fn=None,argv=[],step=False)
   .. autofunction:: exit(all=False)
   .. autofunction:: processArgs(args)
   .. autofunction:: setPrefs(res,save=False)
   .. autofunction:: printall()
   .. autofunction:: printglobals()
   .. autofunction:: printglobalnames()
   .. autofunction:: printconfig()
   .. autofunction:: printdetected()
   .. autofunction:: writable(path)
   .. autofunction:: chdir(fn)
   .. autofunction:: runtime()
   .. autofunction:: startGui(args=[])
   .. autofunction:: isWritable(path)
   .. autofunction:: checkRevision(rev,comp='>=')
   .. autofunction:: writeGeomFile(filename,objects,sep=' ',mode='w')
   .. autofunction:: readGeomFile(filename)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

