.. This may look like plain text, but really is -*- rst -*-

====================================
pyFormex User Meeting 5 (2008-10-31)
====================================


Place and Date
--------------
These are the minutes of the pyFormex User Meeting of Friday October 31, 2008, at the Institute Biomedical Technology (IBiTech), Ghent University, Belgium.


Participants
------------
The following pyFormex developers, users and enthusiasts were present.

- Benedict Verhegghe, chairman
- Matthieu De Beule
- Peter Mortier
- Sofie Van Cauter
- Gianluca De Santis
- Michele Conti
- Tomas Praet, secretary


Apologies
---------
- None


Minutes of the previous meeting
-------------------------------
- No remarks. The minutes are available from the `pyFormex User Meeting page`_.


Agenda and discussion
---------------------
The only item on the agenda is the demonstration of how to run pyFormex on any  computer (even Windows) by means of booting it from a BuMPix Live Linux USB stick. This system was especially developed for students who have only access to Windows computers. 

The BuMPix Linux Live system will boot from an USB memory stick on nearly every computer available. It runs without installation and without ever touching the hard disk of the computer, thus leaving any software on it safely untouched. All programs purely run in memory and disappear when shutting down.
 
The system was developed using the tools of the `Debian Live Project`_. Their website provides a lot more information on how to create and use the Linux Live system.

Benedict starts by giving a spectacular demonstration of the system. The main steps to create and use the system are described hereafter.

* Download the BuMPix image:

  The image file can freely be downloaded from the `BuMPix`_ FTP archive. Currently, the most recent image file is *bumpix-0.4-a7.img*, which is also the one that is demonstrated.

* Write the BuMPix image to a USB stick:

  - The image file size is about 800 MB, so it can conveniently be installed on a USB stick of 1 GB. The remainder can be used as a persistent storage when working with the Live system.

  - Do not simply copy the downloaded image onto your memory stick: it will not run. The image should be written using a low level data copying program. On Linux system this can be done with the command 'dd if=IMAGE of=USBDEV', where IMAGE is the path of the downloaded image file and USBDEV is the designation of the USB device corresponding to your stick. The device name should likely be something like '/dev/sda' or '/dev/sdb', .... CHECK VERY CAREFULLY THAT YOU HAVE THE DRIVE LETTER CORRECT! Or you could end up with wiping your whole hard disk!
    The correct drive letter can be found by issuing the command 'dmesg' after plugging in your stick. (Wait a few seconds to have the harware recognized.) The 'dmesg' log will end with some lines looking like this:
    "... sd 4:0:0:0: [sdb] Blah blah...". In this case, '[sdb]' provides the needed information and you should use '/dev/sdb' in the above 'dd' command. 

    Writing the image to the stick may take five to ten minutes, depending on the speed of your USB system. Be aware that during the linux 'dd' copying process, no feedback is given to the user, which might trick someone into believing that nothing is happening.

  - For those that only have access to a Windows computer, there exist a few utilities to write the image to USB stick. More information can be found on the `Debian Live Project`_ website. Many of our users have reported success with using the *dd for Windows* program. Some users reported failure with *WinRaWrite*.


* Booting the computer from the USB stick:

  - Start or reboot your computer with the USB stick plugged in.

  - Make sure your computer is set to boot from the USB device. In most cases, pressing F12 during booting will take you into a boot selection menu. Select *USB device* or *USB HDD* or *USB Hard Disk* as boot medium (not USB floppy or USB CDROM). If you want to make the choice permanent, so that the computer always boots from the USB-key when it is plugged in, you have to set it in the BIOS (see below).

  - If the USB stick does not show up in your F12 boot menu, you need to set it in the BIOS. In most cases you can reach the BIOS Setup by pressing F2 or DEL during boot. Look for the settings of the Boot sequence and put the USB device on top of the list of bootable hard disks. Then exit the Bios Setup saving your changes. 

  - Some laptops automatically change the BIOS setting and remove the USB entry when you boot without the USB key plugged in. In that case you will have to re-adjust the setting after each non-USB boot of the computer.

  - The boot process will first show the BuMPix boot prompt or (depending on the version) menu. Here, some boot options can be entered or changed, usually to make the system behave well with some odd hardware. Generally, just hit *Enter* to continue the boot process. Most likely everything will startup fine with the default options. If you do encounter problems druing booting, and you do not know how to fix it: ask a wizzard for help.

  - The computer will automatically connect to the network if it is wired to a network with DHCP. Setting up a wireless connection depends on your hardware and wifi parameters, and requires more skills and Linux knowledge.

* Working with the BuMPix Linux Live system:

  - The boot process ends by logging the default user in on the graphical desktop. Currently, the installed desktop system is *KDE*. In future, a *GNOME* variant may be offered also.

  - The default keyboard layout is set to 'us'. Benedict only ever uses Qwerty keyboards. It is the only layout used at boot time, and the easiest to use for programming. For all languages with latin based characters, there currently is no need anymore for other layouts. But since we live in Belgium, and computers come by default with a 'Belgian Azerty' (which is even different from the French Azerty!) keyboard, we had to include the 'be' option. Switch by clicking on the flag button at the bottom right. Other keyboards can be choosen from KDE's keyboard configurator. 
 
  - Only one user is installed, named *user*, with password 'live'. At boot, the system autologins as this user. The hard disk partitions are not mounted by default, to avoid accidentely changing anything. There is no root password and logging in as root has been disabled. The user can get root privileges by using the command *sudo --i*. Only use this if you know what you are doing, and use *exit* to leave the privileged mode as soon as possible.

  - When working with the Live system, all storage is done in RAM memory, and this will be cleared when halting the system. It can therefore be useful to create a second partition on the memory stick, that can be used for persistent storage. How to do this should be explained in the manual. An installer script could be made that automatically creates the second partition.

  - By default the operating system is in Dutch. For now it is only possible to change the language to English, other languages may be added later.

  - The BuMPix Live system contains most general software users need on a computer: e.g browser (Firefox), wordprocessing(OpenOffice), but the system includes many other valuable programs, making BuMPix a full computing environment for most anyone. There is (an evaluation version of) the personal Finite Element pre- and postprocessor *GiD*. And then, off course, there's a recent version of **pyFormex**.

  - Some interesting programs have a start button installed in the lower left panel. The pink 'P' is the pyFormex start button.

* Some suggestions where made of other programs to be included in future versions of the BuMPix Live system:

  - OpenOffice

  - GTS (for surface transformations)

  - A media player

  - A separate *Dutch* directory (to be used for specific Dutch documents, like the Abaqus summer course files)

* Using pyFormex:

  - The use of keywords in the pyFormex examples was mentioned at the previous meeting, but not demonstrated. Classification of the examples based on these keywords is now demonstrated. It is pointed out that a normal user cannot change the examples in an installed pyFormex. So in order to change the parameters of an example, one has to copy the example first to another location. This evenly holds for users of the BuMPix Live system.

  - I18n: The question is raised whether pyFormex menus and documentation should be translated into other languages. The meeting decides not to do so, as users of pyFormex should preferably understand English anyway, since the scripting language contains mostly English terms.


Date of the next meeting
------------------------
The next meeting will be held at IBiTech on Friday December 12, 2008 at 10.00h.


.. Here are the targets referenced in the text

.. _`pyFormex website`: http://pyformex.berlios.de/
.. _`pyFormex home page`: http://pyformex.berlios.de/
.. _`pyFormex user meeting page`: http://pyformex.berlios.de/usermeeting.html
.. _`pyFormex developer site`: http://developer.berlios.de/projects/pyformex/
.. _`pyFormex forums`: http://developer.berlios.de/forum/?group_id=2717
.. _`pyFormex developer forum`: https://developer.berlios.de/forum/forum.php?forum_id=8349
.. _`pyFormex bug tracking`: http://developer.berlios.de/bugs/?group_id=2717
.. _`pyFormex project manager`: mailto:benedict.verhegghe@ugent.be
.. _`UGent digital learning`: https://minerva.ugent.be/main/ssl/login_en.php
.. _`pyFormex news`: http://developer.berlios.de/news/?group_id=2717
.. _`pyformex-announce`: http://developer.berlios.de/mail/?group_id=2717
.. _`IBiTech`: http://www.ibitech.ugent.be/
.. _`BuMPix`: ftp://bumps.ugent.be/pub/bumpix/
.. _`Debian Live Project`: http://wiki.debian.org/DebianLive/Howto/USB/
.. _`WinSCP`: http://winscp.net/eng/index.php

.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

