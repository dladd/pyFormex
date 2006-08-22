#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
##
#
# This file is based on the HelpViewer class from the Qt examples
#
"""A simple HTML help viewer, standalone or embedded in an application.
"""

import qt
import sys,os


class HelpViewer(qt.QMainWindow):
    """HelpViewer is a general purpose help viewer. 
    
    This is a simple hypertext browser intended to view help files.
    It features navigation, history and bookmarks.
    It can be used as a standalone help browser, or embedded in another
    Python application.
    
    It was shaped after the helpviewer example of the Qt documentation.
    It's just about 100 lines of Python code, so don't expect too much.

    It is distributed as part of pyFormex (http://pyformex.berlios.de)
    under the GNU General Public License.
    Authored by Benedict Verhegghe, partly based on TrollTech work.
    """

    def __init__ (self, home, path, histfile=None, bookfile=None, parent=None,
                  name=None, icons=None):
        """Initialize the help window.

        You specify the home page and the path.
        If a history file is specified, it will be read on startup and
        saved when leaving.
        If a bookmarks file is specified, it will be read on startup and
        saved when leaving.
        A list of three icon file names can be specified to override the
        built-in ones. They will be used for back, forward, home.
        """

        qt.QMainWindow.__init__(self, parent, name, qt.Qt.WDestructiveClose)
        self.connect(self, qt.SIGNAL("destroyed()"), self.exit)
        self.pathCombo = qt.QComboBox(0)
        self.selectedURL = None
        self.setCaption("HelpViewer")
        self.setAbout("HelpViewer",self.__doc__)

        self.browser = qt.QTextBrowser(self)
        self.browser.mimeSourceFactory().setFilePath(qt.QStringList(path))
        self.browser.setFrameStyle( qt.QFrame.Panel | qt.QFrame.Sunken)
        self.histfile = histfile
        self.bookfile = bookfile
        self.hist = None
        self.bookm = None
        self.connect(self.browser, qt.SIGNAL("textChanged()"),self.textChanged)
        self.setCentralWidget(self.browser)

        if home:
            self.browser.setSource(home)

        self.connect(self.browser, qt.PYSIGNAL("highlighted(const qt.QString &)"), self.statusBar(), qt.SLOT("message(const qt.QString&)"))
        self.resize(800,600)

        file = qt.QPopupMenu(self)
##        file.insertItem("&New Window", self.newWindow, qt.Qt.CTRL+qt.Qt.Key_N)
        file.insertItem("&Open File", self.openFile, qt.Qt.CTRL+qt.Qt.Key_O)
##        file.insertItem("&Print", self.printit, qt.Qt.CTRL+qt.Qt.Key_P)
        file.insertSeparator()
##        file.insertItem("&Quit", self, qt.SLOT("close()"), qt.Qt.CTRL+qt.Qt.Key_Q)
        file.insertItem("E&xit", self.exit, qt.Qt.CTRL+qt.Qt.Key_X)

        if not icons:
            icons = [ "back.xpm", "forward.xpm", "home.xpm" ]
        icon_back,icon_forward,icon_home = map(qt.QIconSet,map(qt.QPixmap,icons))

        go = qt.QPopupMenu(self)
        self.backwardId = go.insertItem(icon_back,"&Backward",self.browser,
                                        qt.SLOT("backward()"), qt.Qt.CTRL+qt.Qt.Key_Left)
        self.forwardId = go.insertItem(icon_forward,"&Forward",self.browser,
                                       qt.SLOT("forward()"),  qt.Qt.CTRL+qt.Qt.Key_Right)
        go.insertItem(icon_home, "&Home", self.browser, qt.SLOT("home()"))

        help = qt.QPopupMenu(self)
        help.insertItem("&About ...", self.about)
        help.insertItem("About &Qt ...", self.aboutQt)

        self.hist = qt.QPopupMenu(self)
        self.history = self.fillMenu (self.hist,self.histfile)
        qt.QObject.connect(self.hist, qt.SIGNAL("activated(int)"),self.histChosen)

        self.bookmarks = {}
        self.bookm = qt.QPopupMenu(self)
        self.bookm.insertItem("Add Bookmark", self.addBookmark)
        self.bookm.insertSeparator()
        self.bookmarks = self.fillMenu (self.bookm,self.bookfile)
        qt.QObject.connect(self.bookm, qt.SIGNAL("activated(int)"),self.bookmChosen)

        self.menuBar().insertItem("&File", file)
        self.menuBar().insertItem("&Go", go)
        self.menuBar().insertItem("&History", self.hist)
        self.menuBar().insertItem("&Bookmarks", self.bookm)
        self.menuBar().insertSeparator()
        self.menuBar().insertItem("&Help", help)

        self.menuBar().setItemEnabled(self.forwardId, False)
        self.menuBar().setItemEnabled(self.backwardId, False)
        qt.QObject.connect(self.browser, qt.SIGNAL("backwardAvailable(bool)"),
                 self.setBackwardAvailable)
        qt.QObject.connect(self.browser, qt.SIGNAL("forwardAvailable(bool)"),
                 self.setForwardAvailable)

        self.toolbar = qt.QToolBar(self)

        button = qt.QToolButton(icon_back, "Backward", "", self.browser, qt.SLOT("backward()"), self.toolbar)
        qt.QObject.connect(self.browser, qt.SIGNAL("backwardAvailable(bool)"), button, qt.SLOT("setEnabled(bool)"))
        button.setEnabled(False)
        button = qt.QToolButton(icon_forward, "Forward", "", self.browser, qt.SLOT("forward()"), self.toolbar)
        qt.QObject.connect(self.browser, qt.SIGNAL("forwardAvailable(bool)"), button, qt.SLOT("setEnabled(bool)"))
        button.setEnabled(False)
        button = qt.QToolButton(icon_home, "Home", "", self.browser, qt.SLOT("home()"), self.toolbar)

        self.toolbar.addSeparator()

        self.pathCombo = qt.QComboBox(True, self.toolbar)
        qt.QObject.connect(self.pathCombo, qt.PYSIGNAL("activated(const qt.QString &)"), self.pathSelected)
        self.toolbar.setStretchableWidget(self.pathCombo)
        self.setRightJustification(True)
        self.setDockEnabled(qt.Qt.DockLeft, False)
        self.setDockEnabled(qt.Qt.DockRight, False)

        self.pathCombo.insertItem(home)
        self.browser.setFocus()
        self.destroyed = False


    def closeEvent(self,ce):
        print "helpWindow_closeEvent"
        ce.accept()
        self.destroyed = True
        

    def fillMenu(self,menu,filename):
        """Fills a menu with items read from the specified file.

        Returns a directory with the menu id as key and item text as value.
        This will be empty if the file can not be read or None is specified.
        """
        items = {}
        print "Filling from %s" % filename
        for item in self.readFile(filename):
            it = item.rstrip('\n')
            items[menu.insertItem(it)] = it
        return items

                   
    def setBackwardAvailable(self,b):
        self.menuBar().setItemEnabled(self.backwardId, b)

    def setForwardAvailable(self,b):
        self.menuBar().setItemEnabled(self.forwardId, b)

    def textChanged(self):
        tit = qt.QString("HelpViewer - ")
        if self.browser.documentTitle().isNull():
            tit += self.browser.context()
        else:
            tit += self.browser.documentTitle()
        self.setCaption(tit)
        selectedURL = self.browser.context()

        if (not selectedURL.isEmpty() and self.pathCombo):
            exists = False
            for i in range (self.pathCombo.count()):
                if (self.pathCombo.text(i) == selectedURL):
                    exists = True
                    break
            if not exists:
                self.pathCombo.insertItem(selectedURL, 0)
                self.pathCombo.setCurrentItem(0)
                if self.hist:
                    self.history[self.hist.insertItem(selectedURL)] = selectedURL
            else:
                self.pathCombo.setCurrentItem(i)
            selectedURL = qt.QString.null

    def about(self):
        qt.QMessageBox.about(self,self.aboutTitle,self.aboutText)

    def setAbout(self,title,text):
        self.aboutTitle,self.aboutText = title,text

    def aboutQt(self):
        qt.QMessageBox.aboutQt(self, "QBrowser")

    def openFile(self):
        fn = qt.QFileDialog.getOpenFileName(qt.QString.null, qt.QString.null, self)
        if not fn.isEmpty():
            self.browser.setSource(fn)

    def newWindow(self):
        home = str(self.browser.source())
        print ">>%s<<" % home
        path = os.path.dirname(home)
        print path
        HelpViewer(home,path).show()

##    def printit(self):
##        printer = qt.QPrinter()
##        printer.setFullPage(True)
##        if printer.setup(self):
##            p = qt.QPainter(printer)
##            metrics = qt.QPaintDeviceMetrics(p.device())
##            dpix = metrics.logicalDpiX()
##            dpiy = metrics.logicalDpiY()
##            margin = 72 # pt
##            body = qt.QRect(margin*dpix/72, margin*dpiy/72, \
##                       metrics.width()-margin*dpix/72*2, \
##                       metrics.height()-margin*dpiy/72*2)
##            font = qt.QFont("times", 10)
##            richText = qt.QSimpleRichText(self.browser.text(), font, \
##                                      self.browser.context(),     \
##                                      self.browser.styleSheet(),  \
##                                      self.browser.mimeSourceFactory(), \
##                                      body.height())
##            richText.setWidth(p, body.width())
##            view = qt.QRect(body)
##            page = 1
##            while True:
##                richText.draw(p, body.left(), body.top(), view, colorGroup())
##                view.moveBy(0, body.height())
##                p.translate(0 , -body.height())
##                p.setFont(font)
##                p.drawText(view.right() - p.fontMetrics().width(qt.QString.number(page)), \
##                            view.bottom() + p.fontMetrics().ascent() + 5, qt.QString.number(page))
##                if view.top() >= body.top()+richText.height():
##                    break
##                printer.newPage()
##                page += 1

    def pathSelected(self,path):
        self.browser.setSource(path)
        if path not in self.history.values():
            self.history[self.hist.insertItem(path)] = path
        print self.history


    def readFile(self,fn):
        """Read a text file and return the lines in a list"""
        if (fn):
            f = os.path.join(os.getcwd(),str(fn))
            try:
                return file(f,'r').readlines()
            except:
                print "Could not read file %s" % fn
        return []

        
    def writeFile(self,fn,list):
        """Write a list of items to a file"""
        f = file(os.path.join(os.getcwd(),fn),'w')
        for item in list.values():
            f.write(str(item)+'\n')
        f.close()

    def histChosen(self,i):
        if self.history.has_key(i):
            self.browser.setSource(self.history[i])

    def bookmChosen(self,i):
        if self.bookmarks.has_key(i):
            self.browser.setSource(self.bookmarks[i])

    def addBookmark(self):
        self.bookmarks[ self.bookm.insertItem(self.caption()) ] = self.browser.context()

    def exit(self):
        if self.histfile:
            print "Saving history"
            self.writeFile(self.histfile,self.history)
        if self.bookmarks:
            print "Saving bookmarks"
            self.writeFile(self.bookfile,self.bookmarks)
        self.close()



# TEST

if __name__ == "__main__":
    
    def main(args):
        qt.QApplication.setColorSpec(qt.QApplication.ManyColor)
        app = qt.QApplication(sys.argv)
        icons = [ os.path.join(".", "icons", i) for i in [ "prev.xbm","next.xbm","home.xbm" ] ]
        if len(args) > 1:
            home = args[1]
        else:
            home = os.path.join("manual","html","index.html")

        home = os.path.abspath(home)
        path = os.path.dirname(home)
        help = HelpViewer(home, path, ".history", ".bookmarks", None, "help viewer",icons)
 
        if qt.QApplication.desktop().width() > 400 and qt.QApplication.desktop().height() > 500:
            help.show()
        else:
            help.showMaximized()

        qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),app, qt.SLOT("quit()"))
        app.exec_loop()

    main(sys.argv)

# END
