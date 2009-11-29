.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"Use chdir(__file__) instead"'))))))))))))))))), (8, ')'), (4, '')))
.. MATCHING (259, (50, '@'), (287, (1, 'deprecation')))
.. POPPING (260, (259, (50, '@'), (287, (1, 'deprecation')), (7, '('), (329, (330, (303, (304, (305, (306, (307, (309, (310, (311, (312, (313, (314, (315, (316, (317, (3, '"Use chdir(__file__) instead"'))))))))))))))))), (8, ')'), (4, '')))
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



   .. autoclass:: Exit


      Exit objects have the following methods:


   .. autoclass:: ExitAll


      ExitAll objects have the following methods:


   .. autoclass:: ExitSeq


      ExitSeq objects have the following methods:


   .. autoclass:: TimeOut


      TimeOut objects have the following methods:


**Functions defined in the module script**

   .. autofunction:: Globals()
   .. autofunction:: export(dic)
   .. autofunction:: export2(names,values)
   .. autofunction:: forget(names)
   .. autofunction:: rename(oldnames,newnames)
   .. autofunction:: listAll(clas=None,dic=None)
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
   .. autofunction:: formatInfo(F)
   .. autofunction:: printall()
   .. autofunction:: printglobals()
   .. autofunction:: printglobalnames()
   .. autofunction:: printconfig()
   .. autofunction:: printdetected()
   .. autofunction:: writable(path)
   .. autofunction:: chdir(fn)
   .. autofunction:: runtime()
   .. autofunction:: startGui(args=[])
   .. autofunction:: checkRevision(rev,comp='>=')
   .. autofunction:: workHere()

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

