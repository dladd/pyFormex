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

* Starting pyFormex from a USB stick:

  - Benedict starts by giving a spectacular demonstration of the system, which goes as follows:

    + Restart your system with the USB stick inserted.
    + Make sure your computer is set to boot from the USB stick. In most cases, pressing F12 during booting will take you into a boot selection menu. In some cases you may have to go into to BIOS (often by pressing F2 or DEL during boot) to change the boot settings. 
    + Select the USB stick as boot medium. If the USB stick does not show up in your F12 boot menu, you need to set it in the BIOS. The USB stick is recognized as a hard disk. Make sure it is on the top of the list of bootable disks. Some laptops automatically change the BIOS and remove the USB entry when you boot without the USB key plugged in. In that case you will have to re-adjust the setting on each boot of BuMPix.
    + Continue the boot process and you will receive the BuMPix boot prompt screen. Depending on the version, some boot parameters may be added. Generally, just hit enter to boot. Most likely everything will startup fine.
    + If you encounter any problems with booting, and you do not know how to fix it: ask a wizzard for help.

  - The computer should automatically connect to the network if it is wired to a network with DHCP. Setting up a wireless connection depends on your hardware and wifi parameters, and requires more skills and Linux knowledge.

  - By default the operating system will be in Dutch. For now it is only possible to change the language to English, other languages will be added later on. Currently it is planned to include Italian and Danish versions. 

  - The default desktop started is KDE, with a choice between 'us' and 'be' keyboards. Benedict only ever uses Qwerty keyboards, and there is no good reason to use any other keyboard for language with latin based characters. But since we live in Belgium, and computers come by default with a 'Belgian Azerty' (even different from the French!) keyboard, we had to include the 'be' option. Other keyboards can be choosen from KDE's keyboard configurator.

  - Some programs are readily installed, though less than on the CD's that were previously used. Some of those programs are already in the user panel, being a terminal, Firefox, GID (which is an evaluation version of a handy program for Finite Element pre- and postprocessing) and off course a recent version of pyFormex.

* How to get the BuMPix system on the USB stick:

  - The image file can freely be downloaded from the `BuMPix`_ FTP archive. Currently, the most recent image file is bumpix-0.4-a7.img, which is also the one that is enthusiastically demonstrated by Benedict. The total file size is about 800 MB, so it can conveniently be installed on a USB stick of 1 GB.

  - Do not simply copy the downloaded image onto your memory stick: it will not run. The image should be written using a low level data copying program. On Linux system this can be done with the command 'dd if=IMAGE of=USBDEV', where IMAGE is the path of the downloaded image file and USBDEV is the designation of the USB device corresponding to your stick. The device name should likely be something like '/dev/sda' or '/dev/sdb', .... CHECK VERY CAREFULLY THAT YOU HAVE THE DRIVE LETTER CORRECT! Or you could end up with wiping your whole hard disk!
The correct drive letter can be found by issuing the command 'dmesg' after plugging in your stick. (Wait a few seconds to have the harware recognized.) The 'dmesg' log will end with some lines looking like this:
"... sd 4:0:0:0: [sdb] Blah blah...". In this case, '[sdb]' provides the needed information and you should use '/dev/sdb' in the above 'dd' command. 

Writing the image on the stick should only take about six to seven minutes, although the oldest sticks might need up to twenty minutes to do this task. Be aware that during the linux 'dd' copying process, no feedback is given to the user, which might trick someone into believing that nothing is happening.

  - The system was developed using th tools of the `Debian Live Project`_. Their website provides a lot more information on how to write the USB stick, even on Windows systems. Many of our users have reported success by using the *dd for Windows*. Some users reported failure with *WinRaWrite*.


* Working with the BuMPix Linux Live system:

  - Only one user is installed, named *user*, with password 'live'. At boot, the system autologins as this user. The hard disk partations are not mounted by default, to avoi accidentely changing anything. There is no root password and logging in as root has been disabled. The user can get root privileges by using the command *sudo --i*. Only use this if you know what you are doing, and use *exit* to leave the privileged mode as soon as possible.

  - When working with the Live system, all storage is done in RAM memory, and this will be cleared when halting the system. It can therefore be useful to create a second partition on the memory stick, that can be used for persistent storage. How to do this should be explained in the manual.


* Included programs:

- Of course there is pyFormex, but the system includes many other valuable programs, making BuMPix a full computing environment for most anyone.

- Some suggestions where made of other porograms to be included in future versions of the BuMPix Live system:
  - Open Office
  - GTS (for surface transformations)
  - A media player
  - A separate *Dutch* directory (to be used for specific Dutch documents, like the Abaqus summer course files)

* Using pyFormex:
 - The use of keywords in the pyFormex examples was mentioned at the previous meeting, but not demonstrated. Classification of the examples based on these keywords is now demonstrated. It is pointed out that a normal user cannot change the examples in an installed pyFormex. So in order to change the parameters of an example, one has to copy the example first to another location. This evenly holds for users of the BuMPix Live system.

 -I18n: The question is raised whether pyFormex menus and documentation should be translated into other languages. The meeting decides not to do so, as users of pyFormex should preferably understand English anyway, since the scripting language contains mostly English terms.


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

