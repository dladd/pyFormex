#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 09:32:38 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""sendmail.py: a simple program to send an email message

(C) 2008 Benedict Verhegghe (benedict.verhegghe@ugent.be)
I wrote this software in my free time, for my joy, not as a commissioned task.
Any copyright claims made by my employer should therefore be considered void.

Distributed under the GNU General Public License, version 3 or later 
"""

import os,socket,getpass


################### global data ##################################

host = socket.gethostname()
user = getpass.getuser()
mail = os.environ.get('MAIL',"%s@%s" % (user,host))

################### mail access ##################################

    
import smtplib
import email.Message


def message(sender='',to='',cc='',subject='',text=''):
    """Create an email message

    'to' and 'cc' can be lists of email addresses.
    """
    if type(to) is list:
        to = ', '.join(to)
    if type(cc) is list:
        cc = ', '.join(cc)
    message = email.Message.Message()
    message["From"] = sender
    message["To"] = to
    if cc:
        message["Cc"] = cc
    message["Subject"] = subject
    message.set_payload(text)
    return message


def sendmail(message,sender,to,serverURL='localhost'):
    """Send an email message

    'message' is an email message (e.g. returned but message())
    'sender' is a single mail address
    'to' can be a list of addresses
    """
    mailServer = smtplib.SMTP(serverURL)
    mailServer.sendmail(sender,to,message.as_string())
    mailServer.quit()


##################################################################

def input_message(prompt=True):
    print """
    This is Bene's simple mail program, version 0.00001.
    Enter lines of text, end with CTRL-D (on a blank line).
    Include at least one line starting with 'To:'
    Include exactly one line starting with 'Subj:'
    Optionally include a line starting with 'From:'
    Optionally include one or more lines starting with 'CC:'
    All other lines will be the text of your message.
    """
    to = []
    cc = []
    subj = ''
    msg = ''
    sender = ''
    while True:
        try:
            s = raw_input()
            slower = s[:5].lower()
            if slower.startswith('to:'):
                to.append(s[3:])
            elif slower.startswith('cc:'):
                cc.append(s[3:])
            elif slower.startswith('subj:'):
                subj = s[5:]
            elif slower.startswith('from:'):
                sender = s[5:]
            else:
                msg += s+'\n'
            
        except EOFError:
            break
    return to,cc,subj,msg,sender


if __name__ == '__main__':

    to,cc,subj,msg,sender = input_message()
    if not sender:
        sender = mail

    if to and subj and msg and sender:
        msg = message(sender,to,cc,subj,msg)
        print "\n\n    Email message:",msg
        if raw_input('\n    Shall I send the email now? ') == 'y':
            sendmail(msg,sender,to)
            print "Mail has been sent!"
        else:
            print "Mail not sent!"
    else:
        print "Message can not be sent because of missing fields!"
        
# End
