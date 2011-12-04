# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

import pyformex as pf
from gui import menu
from gui.widgets import simpleInputItem as I

import utils
import os

from numpy import *
from formex import *
from gui.draw import *
from gui.colors import *

from subprocess import call


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
access to the host system.
""")


def configure():
    from gui.prefMenu import updateSettings
    from gui.widgets import simpleInputItem as I, groupInputItem as G, tabInputItem as T
    
    dia = None

    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        res = dia.results
        res['_save_'] = save
        if res['_addhost_']:
            res['jobs/hosts'] = pf.cfg['jobs/hosts'] + [ res['_addhost_'] ]
            res['jobs/host'] = res['_addhost_']
        pf.debug(res)
        updateSettings(res)

    def acceptAndSave():
        accept(save=True)

    def autoSettings(keylist):
        return [I(k,pf.cfg[k]) for k in keylist]

    jobs_settings = [
        I('jobs/host',pf.cfg.get('jobs/host','localhost'),text="Host",tooltip="The host machine where your job input/output files are located.",choices=pf.cfg.get('jobs/hosts',['localhost'])), #,buttons=[('Add Host',addHost)]),
        I('jobs/inputdir',pf.cfg.get('jobs/inputdir','bumper/requests'),text="Input directory"),
        I('jobs/outputdir',pf.cfg.get('jobs/outputdir','bumper/results'),text="Output directory"),
        I('_addhost_','',text="New host",tooltip="To set a host name that is not yet in the list of hosts, you can simply fill it in here."),
        ]

    dia = widgets.InputDialog(
        caption='pyFormex Settings',
        store=pf.cfg,
        items=jobs_settings,
        actions=[
            ('Close',close),
            ('Accept and Save',acceptAndSave),
            ('Accept',accept),
        ])
    dia.show()


def getSubdirs(host,userdir):
    """Get a list of all subdirs in userdir on host.

    The host should be a machine where the user has ssh access.
    The userdir is relative to the user's home dir.
    """
    cmd = "ssh %s 'cd %s;ls -F|egrep \".*/\"'" % (host,userdir)
    sta,out = utils.runCommand(cmd,False)
    if sta:
        out = ''
    dirs = out.split('\n')
    dirs = [ j.strip('/') for j in dirs ]
    return dirs


def getFiles(host,userdir,files,targetdir):
    """Copy files from userdir on host to targetdir.

    files is a list of file names.
    """
    files = [ '%s:%s/%s' % (host,userdir.rstrip('/'),f) for f in files ]
    cmd = "scp %s %s" % (' '.join(files),targetdir)
    sta,out = utils.runCommand(cmd)
    return sta==0


def remoteCommand(host=None,command=None):
    """Execute a remote command.

    host: the hostname where the command is executed
    command: the command line
    """
    if host is None or command is None:
        res = askItems(
            [ I('host',choices=['bumpfs','bumpfs2','--other--']),
              I('other','',text='Other host name'),
              I('command','hostname'),
              ],
            enablers = [('host','--other--','other')],
            )

    if res:
        host = res['host']
        if host == '--other--':
            host = res['other']
        command = res['command']

    if host and command:
        cmd = "ssh %s '%s'" % (host,command)
        sta,out = utils.runCommand(cmd)
        message(out)
        

def submitToCluster(filename=None):
    """Submit an Abaqus job to the cluster."""
    if not filename:
        filename = askFilename(pf.cfg['workdir'],filter="Abaqus input files (*.inp)",exist=True)
    if filename:
        if not filename.endswith('.inp'):
            filename += '.inp'
        jobname = os.path.basename(filename)[:-4]
        res = askItems([
            ('ncpus',4,{'text':'Number of cpus','min':1,'max':1024}),
            ('postabq',False,{'text':'Run postabq on the results?'}),
            ])
        if res:
            reqtxt = 'cpus=%s\n' % res['ncpus']
            if res['postabq']:
                reqtxt += 'postproc=postabq\n'
            host = pf.cfg.get('jobs/host','mecaflix')
            reqdir = pf.cfg.get('jobs/inputdir','bumper/requests')
            cmd = "scp %s %s:%s" % (filename,host,reqdir)
            ret = call(['scp',filename,'%s:%s' % (host,reqdir)])
            print ret
            ret = call(['ssh',host,"echo '%s' > %s/%s.request" % (reqtxt,reqdir,jobname)])
            print ret
        

def killClusterJob(jobname=None):
    """Kill a job to the cluster."""
    res = askItems([('jobname','')])
    if res:
        jobname = res['jobname']
        host = pf.cfg.get('jobs/host','mecaflix')
        reqdir = pf.cfg.get('jobs/inputdir','bumper/requests')
        cmd = "touch %s/%s.kill" % (reqdir,jobname)
        print host
        print cmd
        ret = call(['ssh',host,"%s" % cmd])
        print ret
       

the_host = None
the_userdir = None
the_jobnames = None
the_jobname = None


def checkResultsOnServer(host=None,userdir=None):
    """Get a list of job results from the cluster.

    Specify userdir='bumper/running' to get a list of running jobs.
    """
    global the_host,the_userdir,the_jobnames
    if host is None or userdir is None:
        res = askItems([
            ('host',None,'select',{'choices':['bumpfs','bumpfs2','other']}),
            ('other','',{'text':'Other host name'}),
            ('status',None,'select',{'choices':['results','running','custom']}),
            ('userdir','bumper/results/',{'text':'Custom user directory'}),
            ])
        if not res:
            return
        host = res['host']
        if host == 'other':
            host = res['other']
        status = res['status']
        if status in ['results','running']:
            userdir = 'bumper/%s/' % status
        else:
            userdir = res['userdir']
        
    jobnames = getSubdirs(host,userdir)
    if jobnames:
        the_host = host
        the_userdir = userdir
        the_jobnames = jobnames
    else:
        the_host = None
        the_userdir = None
        the_jobnames = None
    pf.message(the_jobnames)
       

def changeTargetDir(fn):
    from gui import draw
    return draw.askDirname(fn)
    

def getResultsFromServer(jobname=None,targetdir=None,ext=['.fil']):
    """Get results back from cluster."""
    global the_jobname
    print "getRESULTS"
    if targetdir is None:
        targetdir = pf.cfg['workdir']
    if jobname is None:
        if the_jobnames is None:
            jobname_input = [
                ('host',pf.cfg['jobs/host']),
                ('userdir','bumper/results'),
                ('jobname',''),
                ]
        else:
            jobname_input = [
                dict(name='jobname',value=the_jobname,choices=the_jobnames)
                ]

        print jobname_input
        res = askItems(jobname_input + [
            dict(name='target dir',value=targetdir,itemtype='button',func=changeTargetDir),
            ('.fil',True),
            ('.post',True),
            ('_post.py',False),
            ])
        if res:
            host = res.get('host',the_host)
            userdir = res.get('userdir',the_userdir)
            jobname = res['jobname']
            targetdir = res['target dir']
            ext = [ e for e in ['.fil','.post','_post.py'] if res[e] ]
    if jobname and ext:
        files = [ '%s%s' % (jobname,e) for e in ext ]
        userdir = "%s/%s" % (userdir,jobname)
        if getFiles(host,userdir,files,targetdir):
            the_jobname = jobname

        

####################################################################
######### MENU #############

_menu = 'Jobs'

def create_menu():
    """Create the Jobs menu."""
    MenuData = [
        ("&About",about),
        ("&Configure Job Plugin",configure),
        ("&Submit Abaqus Job",submitToCluster),
        ("&Kill Cluster Job",killClusterJob),
        ("&List available results on server",checkResultsOnServer),
        ("&Get results from server",getResultsFromServer),
        ("&Execute remote command",remoteCommand),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help')


def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the menu."""
    pf.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    reload_menu()

# End

