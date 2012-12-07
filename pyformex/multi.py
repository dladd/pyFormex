# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
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
#
"""Framework for multi-processing in pyFormex

This module contains some functions to perform multiprocessing inside
pyFormex in a unified way.
   
"""
from __future__ import print_function

from multiprocessing import Pool,cpu_count


def dofunc(arg):
    """Helper function for the multitask function.

    It expects a tuple with (function,args) as single argument.
    """
    func,args = arg
    return func(*args)


def multitask(tasks,nproc=-1):
    """Perform tasks in parallel.

    Runs a number of tasks in parallel over a number of subprocesses.

    Parameters:
    
    - `tasks` : a list of (function,args) tuples, where function is a
      callable and args is a tuple with the arguments to be passed to the
      function.
    - ` nproc`: the number of subprocesses to be started. This may be
      different from the number of tasks to run: processes finishing a
      task will pick up a next one. There is no benefit in starting more
      processes than the number of tasks or the number of processing units
      available. The default will set `nproc` to the minimum of these two
      values.
    """
    if nproc < 0:
        nproc = min(len(tasks),cpu_count())
        
    pool = Pool(nproc)
    res = pool.map(dofunc,tasks)
    return res


### Following is an alternative using Queues

def worker(input, output):
    """Helper function for the multitask function.

    This is the function executed by any of the processes started
    by the multitask function. It takes tuples (function,args) from
    the input queue, computes the results of the call function(args),
    and pushes these results on the output queue. 

    Parameters:

    - `input`: Queue holding the tasks to be performed
    - `output`: Queue where the results are to be delivered
    """
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)


def multitask2(tasks,nproc=-1):
    """Perform tasks in parallel.

    Runs a number of tasks in parallel over a number of subprocesses.

    Parameters:
    
    - `tasks` : a list of (function,args) tuples, where function is a
      callable and args is a tuple with the arguments to be passed to the
      function.
    - ` nproc`: the number of subprocesses to be started. This may be
      different from the number of tasks to run: processes finishing a
      task will pick up a next one. There is no benefit in starting more
      processes than the number of tasks or the number of processing units
      available. The default will set `nproc` to the minimum of these two
      values.
    """
    if nproc < 0:
        nproc = min(len(tasks),cpu_count())
    
    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in tasks:
        task_queue.put(task)

    # Start worker processes
    for i in range(nproc):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get results
    res = [ done_queue.get() for i in range(len(tasks)) ]
    
    # Tell child processes to stop
    for i in range(nproc):
        task_queue.put('STOP')

    # Return result
    return res


# End
