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
There's basically only one item on the agenda, being the demonstration on how to run pyFormex on a Windows computer by means of booting from a BuMPix Live Linux USB stick. This system was especially developed for the students, who for the most part use windows computers.

* Starting pyFormex from a USB stick:

  - Benedict starts by giving a spectacular demonstration of the system, which goes as follows:

    + Restart your system with the USB stick inserted.
    + During the booting of the computer, call the boot screen, in this case by pressing F12. Other computers might require pressing the Esc key.
    + Select the USB stick as drive to boot from.
    + Another boot screen will appear on which you can choose to press either F1 (which allows you to set the parameters for the boot) or ENTER (which will launch the Linux system under which pyFormex runs).

  - The computer should automatically connect to the network if it is connected by ethernet cable and if it is a DHCP network. Whether a wireless connection can be started automatically remains uncertain.

  - By default the operating system will be in Dutch. For now it is only possible to change the language to English, other languages will be added later on. It is noted that there is special interest for Italian and Danish versions. Also, there are two keyboard configurations implemented: `be' and `us'. Benedict unintentionally demonstrates that those two settings are not enough to cover all commonly used keyboards.

  - Some programs are readily installed, though less than on the CD's that were previously used. Some of those programs are already in the user panel, being a terminal, Firefox, GID (which is an evaluation version of a handy program for post processing) and off course a recent version of pyFormex.

* How to get the BuMPix system on the USB stick:

  - The image file can freely be downloaded from `BuMPix`_. Currently, the most recent image file is bumpix-0.4-a7.img, which is also the one that is enthusiastically demonstrated by Benedict. The total file size is about 800 MB, so it can conveniently be installed on a USB stick of 1 GB. Remarkably, writing the system on a stick of 1 GB seems to leave around 400 MB of free space, which clearly both startled and intrigued the demonstrator.

  - Simply copying the image to your memory stick does not suffice to run the program. It should in fact be copied by a low level program that can write the image byte by byte to the USB stick. This in turn creates a certain risk, because such programs do not care for the data that are already present on the place you are writing to, it simply overwrites them without any warning at all. Accidently choosing the wrong the destination for the image can even render your hard drive useless. Also, any data present on the memory stick will be deleted upon installing the system. It is suggested that a procedure could be written that checks the size of the station where the image is written to, this for example could only allow installation on disks with a volume between 800 MB and 4 GB, thus effectively preventing accidental overwrite of any hard disk.

  - On Linux system the copying can be carried out by the command line "dd if=bumpix-version.img of=USB DEV". Special attention should be paid to whether you are writing to the correct drive. One can check the location of the USB stick with the command dmsg. It is good practice to always check the USB stick's location this way, as the drive name of the USB stick can potentially change, even if you use the same computer.

  - The system was developed based on technicalities that can be found on the `Debian Live Project`_. Information on how to write the USB stick with windows is provided there, though some students reported that the suggested program *WinRaWrite* didn't do the job, possibly because they used incorrect options.

  - Writing the image on the stick should only take about six to seven minutes, although the oldest sticks might need up to twenty minutes to do this task. It is important to notice that during the entire writing sequence there is no feed back at all on the screen, which might trick someone into believing that nothing is happening. Because of this it was stated that students should be clearly instructed not to turn off or restart their computer until the whole process is finished. Not doing so can ruin the memory stick.

* Working with the Linux system:

  - Only one user is installed, named *User*. By default the root permissions are disabled, so that the user can't access the hard disk. This serves as a protection against accidental changing of data on the hard disk. Root permissions can be acquired by the command line *rude --i*. In this way the user can do more than just the ´normal tasks´.

  - Some programs are readily installed on the system, including several basic text editors like KWrite. Benedict also points out that the education programs can be very interesting for the students. A statement that was supported by a fascinating demonstration on the table of Mendeljev.

  - It can be useful to create a partition on the memory stick. This way, data can be made available for use on other systems. How to do this should be explained in the manual.

* Included programs:

The participants were given the opportunity to compose a wish list for preinstalled programs and features. The suggestions included:
  - Open Office
  - GTS (for surface transformations)
  - A media player
  - A separate *Dutch* directory (to be used for specific Dutch documents, like the Abaqus summer course files)

* Using pyFormex

It is noted that the use of keywords in the pyFormex examples was mentioned but not demonstrated during the last meeting. Classification of the examples based on these keywords is demonstrated. It is pointed out that one cannot change the examples that are provided in pyFormex. So in order of playing with the parameters of an example, one has to copy the example first to another location, this is also demonstrated by Benedict. The map *My Scripts* can for example be used for copying the examples to. Add-ons to pyFormex and installation of additional programs can be done by downloading directly in the operating system onto the memory stick.


Varia
-----
  - The participants are asked for their opinion regarding additional language support. More in particular, it is suggested that the menus of the GUI could be made available in other languages. The participants agreed on not doing so. People who use pyFormex should preferably understand English anyway, as the scripting is always done in English.


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

