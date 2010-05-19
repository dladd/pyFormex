.. $Id$  -*- rst -*-
.. pyformex reference manual --- utils
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-utils:

:mod:`utils` --- A collection of miscellaneous utility functions.
=================================================================

.. automodule:: utils
   :synopsis: A collection of miscellaneous utility functions.



   .. autoclass:: NameSequence


      NameSequence objects have the following methods:

      .. automethod:: next()
      .. automethod:: peek()
      .. automethod:: glob()

**Functions defined in the module utils**

   .. autofunction:: checkVersion(name,version,external=False)
   .. autofunction:: hasModule(name,check=False)
   .. autofunction:: hasExternal(name)
   .. autofunction:: checkModule(name=None)
   .. autofunction:: checkExternal(name=None,command=None,answer=None)
   .. autofunction:: reportDetected()
   .. autofunction:: procInfo(title)
   .. autofunction:: strNorm(s)
   .. autofunction:: prefixFiles(prefix,files)
   .. autofunction:: matchMany(regexps,target)
   .. autofunction:: matchCount(regexps,target)
   .. autofunction:: matchAny(regexps,target)
   .. autofunction:: matchNone(regexps,target)
   .. autofunction:: matchAll(regexps,target)
   .. autofunction:: listTree(path,listdirs=True,topdown=True,sorted=False,excludedirs=[],excludefiles=[],includedirs=[],includefiles=[])
   .. autofunction:: removeTree(path,top=True)
   .. autofunction:: setSaneLocale(localestring='')
   .. autofunction:: dos2unix(infile,outfile=None)
   .. autofunction:: unix2dos(infile,outfile=None)
   .. autofunction:: all_image_extensions()
   .. autofunction:: fileDescription(ftype)
   .. autofunction:: findIcon(name)
   .. autofunction:: projectName(fn)
   .. autofunction:: splitme(s)
   .. autofunction:: mergeme(s1,s2)
   .. autofunction:: mtime(fn)
   .. autofunction:: timeEval(s,glob=None)
   .. autofunction:: countLines(fn)
   .. autofunction:: runCommand(cmd,RaiseError=True,quiet=False)
   .. autofunction:: spawn(cmd)
   .. autofunction:: killProcesses(pids,signal)
   .. autofunction:: changeExt(fn,ext)
   .. autofunction:: tildeExpand(fn)
   .. autofunction:: isPyFormex(filename)
   .. autofunction:: splitEndDigits(s)
   .. autofunction:: splitStartDigits(s)
   .. autofunction:: prefixDict(d,prefix='')
   .. autofunction:: subDict(d,prefix='')
   .. autofunction:: stuur(x,xval,yval,exp=2.5)
   .. autofunction:: interrogate(item)
   .. autofunction:: deprecation(message)
   .. autofunction:: deprecated(replacement)
   .. autofunction:: functionWasRenamed(replacement,text=None)
   .. autofunction:: functionBecameMethod(replacement)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

