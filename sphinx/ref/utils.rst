.. $Id$  -*- rst -*-
.. pyformex reference manual --- utils
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-utils:

:mod:`utils` --- A collection of miscellaneous utility functions.
=================================================================

.. automodule:: utils
   :synopsis: A collection of miscellaneous utility functions.



   .. autoclass:: NameSequence
      :members: next,peek,glob

**Functions defined in the module utils**

   .. autofunction:: checkVersion(name,version,external=???)
   .. autofunction:: hasModule(name,check=???)
   .. autofunction:: hasExternal(name)
   .. autofunction:: checkModule(name=???)
   .. autofunction:: checkExternal(name=???,command=???,answer=???)
   .. autofunction:: reportDetected()
   .. autofunction:: procInfo(title)
   .. autofunction:: strNorm(s)
   .. autofunction:: prefixFiles(prefix,files)
   .. autofunction:: matchMany(regexps,target)
   .. autofunction:: matchCount(regexps,target)
   .. autofunction:: matchAny(regexps,target)
   .. autofunction:: matchNone(regexps,target)
   .. autofunction:: matchAll(regexps,target)
   .. autofunction:: listTree(path,listdirs=???,topdown=???,sorted=???,excludedirs=???,excludefiles=???,includedirs=???,includefiles=???)
   .. autofunction:: removeTree(path,top=???)
   .. autofunction:: setSaneLocale(localestring=???)
   .. autofunction:: dos2unix(infile,outfile=???)
   .. autofunction:: unix2dos(infile,outfile=???)
   .. autofunction:: all_image_extensions()
   .. autofunction:: fileDescription(ftype)
   .. autofunction:: findIcon(name)
   .. autofunction:: projectName(fn)
   .. autofunction:: splitme(s)
   .. autofunction:: mergeme(s1,s2)
   .. autofunction:: mtime(fn)
   .. autofunction:: timeEval(s,glob=???)
   .. autofunction:: countLines(fn)
   .. autofunction:: runCommand(cmd,RaiseError=???,quiet=???)
   .. autofunction:: spawn(cmd)
   .. autofunction:: killProcesses(pids,signal)
   .. autofunction:: changeExt(fn,ext)
   .. autofunction:: tildeExpand(fn)
   .. autofunction:: isPyFormex(filename)
   .. autofunction:: splitEndDigits(s)
   .. autofunction:: splitStartDigits(s)
   .. autofunction:: prefixDict(d,prefix=???)
   .. autofunction:: subDict(d,prefix=???)
   .. autofunction:: stuur(x,xval,yval,exp=???)
   .. autofunction:: interrogate(item)
   .. autofunction:: deprecation(message)
   .. autofunction:: deprecated(replacement)
   .. autofunction:: functionWasRenamed(replacement,text=???)
   .. autofunction:: functionBecameMethod(replacement)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

