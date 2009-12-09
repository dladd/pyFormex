#!/usr/bin/env python pyformex.py
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Tue Dec  8 12:25:08 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

import pyformex as GD
from gui import widgets
import utils
import os

from numpy import *
from formex import *
from gui.draw import *
from gui.colors import *


def about():
    showInfo("""Jobs.py

This is pyFormex plugin allowing the user to
- submit computational jobs to a cluster,
- check available job results on a remote host,
- copy job results to the local workstation,
- execute a command on the remote host.

While primarily intended for use with the BuMPer cluster at
IBiTech-bioMMeda, we have made this plugin available to the
general public, as an example of how to integrate external
commands and hosts into a pyFormex menu.

In order for these commands to work, you need to have `ssh`
access to the servers.
""")


def getSubdirs(server,userdir):
    """Get a list of all subdirs in userdir on server.

    The server should be a machine where the user has ssh access.
    The userdir is relative to the user's home dir.
    """
    cmd = "ssh %s 'cd %s;ls -F|egrep \".*/\"'" % (server,userdir)
    sta,out = utils.runCommand(cmd,False)
    if sta:
        out = ''
    dirs = out.split('\n')
    dirs = [ j.strip('/') for j in dirs ]
    return dirs


def getFiles(server,userdir,files,targetdir):
    """Copy files from userdir on server to targetdir.

    files is a list of file names.
    """
    print(server,userdir,files,targetdir)
    files = [ '%s:%s/%s' % (server,userdir,f) for f in files ]
    cmd = "scp %s %s/" % (' '.join(files),targetdir)
    sta,out = utils.runCommand(cmd,False)
    return sta==0


def remoteCommand(server=None,command=None):
    """Execute a remote command."""
    if server is None or command is None:
        res = askItems([
            ('server',None,'radio',{'select':['bumpfs','bumpfs2','other']}),
            ('other','',{'text':'Other server name'}),
            ('command',''),
            ])
    if res:
        server = res['server']
        if server == 'other':
            server = res['other']
        command = res['command']

    if server and command:
        cmd = "ssh %s '%s'" % (server,command)
        sta,out = utils.runCommand(cmd)
        message(out)
        

def submitToCluster(filename=None):
    """Submit an Abaqus job to the cluster."""
    if not filename:
        filename = askFilename(GD.cfg['workdir'],filter="Abaqus input files (*.inp)",exist=True)
    if filename:
        if not filename.endswith('.inp'):
            filename += '.inp'
        jobname = os.path.basename(filename)[:-4]
        res = askItems([
            ('ncpus',4,{'text':'Number of cpus','min':1,'max':1024}),
            ('postabq',False,{'text':'Run postabq on the results?'}),
            ])
        reqtxt = 'cpus=%s\n' % res['ncpus']
        if res['postabq']:
            reqtxt += 'postproc=postabq\n'
        host = GD.cfg.get('jobs/host','mecaflix')
        reqdir = GD.cfg.get('jobs/requests','bumper/requests')
        cmd = "scp %s %s:%s" % (filename,host,reqdir)
        from subprocess import call
        ret = call(['scp',filename,'%s:%s' % (host,reqdir)])
        print ret
        ret = call(['ssh',host,"echo '%s' > %s/%s.request" % (reqtxt,reqdir,jobname)])
        print ret


the_server = None
the_userdir = None
the_jobnames = None
the_jobname = None


def checkResultsOnServer(server=None,userdir=None):
    """Get a list of job results from the cluster.

    Specify userdir='bumper/running' to get a list of running jobs.
    """
    global the_server,the_userdir,the_jobnames
    if server is None or userdir is None:
        res = askItems([
            ('server',None,'select',{'choices':['bumpfs','bumpfs2','other']}),
            ('other','',{'text':'Other server name'}),
            ('status',None,'select',{'choices':['results','running','custom']}),
            ('userdir','bumper/results/',{'text':'Custom user directory'}),
            ])
        if not res:
            return
        server = res['server']
        if server == 'other':
            server = res['other']
        status = res['status']
        if status in ['results','running']:
            userdir = 'bumper/%s/' % status
        else:
            userdir = res['userdir']
        
    jobnames = getSubdirs(server,userdir)
    if jobnames:
        the_server = server
        the_userdir = userdir
        the_jobnames = jobnames
    else:
        the_server = None
        the_userdir = None
        the_jobnames = None
    GD.message(the_jobnames)
        
    

def getResultsFromServer(jobname=None,targetdir=None,ext=['.fil']):
    """Get results back from cluster."""
    global the_jobname
    if targetdir is None:
        targetdir = GD.cfg['workdir']
    if jobname is None:
        if the_jobnames is None:
            jobname_input = [('server','mecaflix'),
                             ('userdir','bumper/results'),
                             ('jobname',''),
                             ]
        else:
            jobname_input = [('jobname',the_jobname,'select',the_jobnames)]

        res = askItems(jobname_input + [('target dir',targetdir),
                                        ('.fil',True),
                                        ('.post',True),
                                        ('_post.py',False),
                                        ])
        if res:
            server = res.get('server',the_server)
            userdir = res.get('userdir',the_userdir)
            jobname = res['jobname']
            targetdir = res['target dir']
            ext = [ e for e in ['.fil','.post','_post.py'] if res[e] ]
    if jobname and ext:
        files = [ '%s%s' % (jobname,e) for e in ext ]
        userdir = "%s/%s" % (userdir,jobname)
        if getFiles(server,userdir,files,targetdir):
            the_jobname = jobname

        

####################################################################
######### MENU #############

def create_menu():
    """Create the Jobs menu."""
    MenuData = [
        ("&About",about),
        ("&Submit Abaqus Job",submitToCluster),
        ("&Check result cases on server",checkResultsOnServer),
        ("&Get results from server",getResultsFromServer),
        ("&Execute remote command",remoteCommand),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu('Jobs',items=MenuData,parent=GD.GUI.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not GD.GUI.menu.item('Jobs'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = GD.GUI.menu.item('Jobs')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()

if __name__ == "draw":
    reload_menu()

# End

