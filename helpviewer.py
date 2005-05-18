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

from qt import *
import sys,os,string

def tr(s):
    return s

class HelpViewer(QMainWindow):
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

    def __init__ (self, home, path, histfile=None, bookfile=None, parent=None, name=None, icons=None):
        """Initialize the help window.

        You specify the home page and the path.
        If a history file is specified, it will be read on startup and
        saved when leaving.
        If a bookmarks file is specified, it will be read on startup and
        saved when leaving.
        A list of three icon file names can be specified to override the
        built-in ones. They will be used for back, forward, home.
        """

        QMainWindow.__init__(self, parent, name, Qt.WDestructiveClose)
        self.pathCombo = QComboBox(0)
        self.selectedURL = None
        self.setCaption("HelpViewer")
        self.setAbout("HelpViewer",self.__doc__)

        self.browser = QTextBrowser(self)
        self.browser.mimeSourceFactory().setFilePath(QStringList(path))
        self.browser.setFrameStyle( QFrame.Panel | QFrame.Sunken)
        self.histfile = histfile
        self.bookfile = bookfile
        self.hist = None
        self.bookm = None
        self.connect(self.browser, SIGNAL("textChanged()"),self.textChanged)
        self.setCentralWidget(self.browser)

        if home:
            self.browser.setSource(home)

        self.connect(self.browser, SIGNAL("highlighted(const QString &)"), 
             self.statusBar(), SLOT("message(const QString&)"))
        self.resize(800,600)

        file = QPopupMenu(self)
        file.insertItem(tr("&New Window"), self.newWindow, Qt.CTRL+Qt.Key_N)
        file.insertItem(tr("&Open File"), self.openFile, Qt.CTRL+Qt.Key_O)
##        file.insertItem(tr("&Print"), self.printit, Qt.CTRL+Qt.Key_P)
        file.insertSeparator()
        file.insertItem(tr("&Quit"), self, SLOT("close()"), Qt.CTRL+Qt.Key_Q)
        file.insertItem(tr("E&xit"), self.exit, Qt.CTRL+Qt.Key_X)

        # The same three icons are used twice each.
        if icons:
            icon_back,icon_forward,icon_home = map(QIconSet,map(QPixmap,icons))
        else:
            icon_back = QIconSet(QPixmap([
                "20 20 2 1",
                "X c black",
                ". c None",
                "....................",
                "....................",
                ".............X......",
                "............xX......",
                "...........XXX......",
                "..........XXXX......",
                ".........XXXXX......",
                "........XXXXXX......",
                ".......XXXXXXX......",
                "......XXXXXXXX......",
                "......XXXXXXXX......",
                ".......XXXXXXX......",
                "........XXXXXX......",
                ".........XXXXX......",
                "..........XXXX......",
                "...........XXX......",
                "............XX......",
                ".............X......",
                "....................",
                "...................."
                ]))
            icon_forward = QIconSet(QPixmap([
                "20 20 2 1",
                "X c black",
                ". c None",
                "....................",
                "....................",
                "......X.............",
                "......XX............",
                "......XXX...........",
                "......XXXX..........",
                "......XXXXX.........",
                "......XXXXXX........",
                "......XXXXXXX.......",
                "......XXXXXXXX......",
                "......XXXXXXXX......",
                "......XXXXXXX.......",
                "......XXXXXX........",
                "......XXXXX.........",
                "......XXXX..........",
                "......XXX...........",
                "......XX............",
                "......X.............",
                "....................",
                "...................."
                ]))
            icon_home = QIconSet(QPixmap(
                ["20 20 2 1",
                 "X c black",
                 ". c None",
                 "....................",
                 "....XX..............",
                 "....XX...XX.........",
                 "....XX.XXXXXX.......",
                 "....XXXXXXXXXXX.....",
                 "...XXXXXXXXXXXXXX...",
                 "..XXXXXXXXXXXXXXXX..",
                 "....X..........X....",
                 "....X......XXX.x....",
                 "....X......X.X.X....",
                 "....X......XXX.X....",
                 "....X..........X....",
                 "....X...XXX....X....",
                 "....X...X.X....X....",
                 "....X...X.X....X....",
                 "....X...X.X....X....",
                 "....X...X.X....X....",
                 "....X...X.X....X....",
                 "....................",
                 "...................."
                 ]))

        go = QPopupMenu(self)
        self.backwardId = go.insertItem(icon_back, \
            tr("&Backward"), self.browser, SLOT("backward()"), Qt.CTRL+Qt.Key_Left)
        self.forwardId = go.insertItem(icon_forward, \
            tr("&Forward"), self.browser, SLOT("forward()"),  Qt.CTRL+Qt.Key_Right)
        go.insertItem(icon_home, tr("&Home"), self.browser, SLOT("home()"))

        help = QPopupMenu(self)
        help.insertItem(tr("&About ..."), self.about)
        help.insertItem(tr("About &Qt ..."), self.aboutQt)

        self.hist = QPopupMenu(self)
        self.history = self.fillMenu (self.hist,self.histfile)
        QObject.connect(self.hist, SIGNAL("activated(int)"),self.histChosen)

        self.bookmarks = {}
        self.bookm = QPopupMenu(self)
        self.bookm.insertItem(tr("Add Bookmark"), self.addBookmark)
        self.bookm.insertSeparator()
        self.bookmarks = self.fillMenu (self.bookm,self.bookfile)
        QObject.connect(self.bookm, SIGNAL("activated(int)"),self.bookmChosen)

        self.menuBar().insertItem(tr("&File"), file)
        self.menuBar().insertItem(tr("&Go"), go)
        self.menuBar().insertItem(tr("History"), self.hist)
        self.menuBar().insertItem(tr("Bookmarks"), self.bookm)
        self.menuBar().insertSeparator()
        self.menuBar().insertItem(tr("&Help"), help)

        self.menuBar().setItemEnabled(self.forwardId, False)
        self.menuBar().setItemEnabled(self.backwardId, False)
        QObject.connect(self.browser, SIGNAL("backwardAvailable(bool)"),
                 self.setBackwardAvailable)
        QObject.connect(self.browser, SIGNAL("forwardAvailable(bool)"),
                 self.setForwardAvailable)

        self.toolbar = QToolBar(self)

        button = QToolButton(icon_back, tr("Backward"), "", self.browser, SLOT("backward()"), self.toolbar)
        QObject.connect(self.browser, SIGNAL("backwardAvailable(bool)"), button, SLOT("setEnabled(bool)"))
        button.setEnabled(False)
        button = QToolButton(icon_forward, tr("Forward"), "", self.browser, SLOT("forward()"), self.toolbar)
        QObject.connect(self.browser, SIGNAL("forwardAvailable(bool)"), button, SLOT("setEnabled(bool)"))
        button.setEnabled(False)
        button = QToolButton(icon_home, tr("Home"), "", self.browser, SLOT("home()"), self.toolbar)

        self.toolbar.addSeparator()

        self.pathCombo = QComboBox(True, self.toolbar)
        QObject.connect(self.pathCombo, SIGNAL("activated(const QString &)"), \
                 self.pathSelected)
        self.toolbar.setStretchableWidget(self.pathCombo)
        self.setRightJustification(True)
        self.setDockEnabled(Qt.DockLeft, False)
        self.setDockEnabled(Qt.DockRight, False)

        self.pathCombo.insertItem(home)
        self.browser.setFocus()

    def fillMenu(self,menu,filename):
        """Fills a menu with items read from the specified file.

        Returns a directory with the menu id as key and item text as value.
        This will be empty if the file can not be read or None is specified.
        """
        items = {}
        if filename:
            try:
                for item in self.readFile(filename):
                    it = string.rstrip(item,"\n")
                    items[menu.insertItem(it)] = it
            except:
                print "Could not read file %s"%filename
        return items
                   
    def setBackwardAvailable(self,b):
        self.menuBar().setItemEnabled(self.backwardId, b)

    def setForwardAvailable(self,b):
        self.menuBar().setItemEnabled(self.forwardId, b)

    def textChanged(self):
        tit = QString("HelpViewer - ")
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
            selectedURL = QString.null

    def about(self):
        QMessageBox.about(self,self.aboutTitle,self.aboutText)

    def setAbout(self,title,text):
        self.aboutTitle,self.aboutText = title,text

    def aboutQt(self):
        QMessageBox.aboutQt(self, "QBrowser")

    def openFile(self):
        fn = QFileDialog.getOpenFileName(QString.null, QString.null, self)
        if not fn.isEmpty():
            self.browser.setSource(fn)

    def newWindow(self):
        HelpViewer(self.browser.source(), ".").show()

##    def printit(self):
##        printer = QPrinter()
##        printer.setFullPage(True)
##        if printer.setup(self):
##            p = QPainter(printer)
##            metrics = QPaintDeviceMetrics(p.device())
##            dpix = metrics.logicalDpiX()
##            dpiy = metrics.logicalDpiY()
##            margin = 72 # pt
##            body = QRect(margin*dpix/72, margin*dpiy/72, \
##                       metrics.width()-margin*dpix/72*2, \
##                       metrics.height()-margin*dpiy/72*2)
##            font = QFont("times", 10)
##            richText = QSimpleRichText(self.browser.text(), font, \
##                                      self.browser.context(),     \
##                                      self.browser.styleSheet(),  \
##                                      self.browser.mimeSourceFactory(), \
##                                      body.height())
##            richText.setWidth(p, body.width())
##            view = QRect(body)
##            page = 1
##            while True:
##                richText.draw(p, body.left(), body.top(), view, colorGroup())
##                view.moveBy(0, body.height())
##                p.translate(0 , -body.height())
##                p.setFont(font)
##                p.drawText(view.right() - p.fontMetrics().width(QString.number(page)), \
##                            view.bottom() + p.fontMetrics().ascent() + 5, QString.number(page))
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
        f = os.path.join(os.getcwd(),fn)
        if os.path.isfile(f): 
            return file(f,'r').readlines()
        else:
            return []
        
    def writeFile(self,fn,list):
        """Write a list of items to a file"""
        print list
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
            self.writeFile(self.histfile,self.history)
        if self.bookmarks:
            self.writeFile(self.bookfile,self.bookmarks)
        self.close()


#TEST

if __name__ == "__main__":
    
    def main(args):
        QApplication.setColorSpec(QApplication.ManyColor)
        app = QApplication(sys.argv)
        icons = None
        if len(args) > 1:
            home = args[1]
        elif os.environ.has_key("QTDIR"):
            qtdir = os.environ["QTDIR"]
            home = qtdir + "/doc/html/index.html"
            icondir = qtdir + "/examples/helpviewer/"
            if os.path.isdir(icondir):
                icons = [ icondir + icon for icon in [ "back.xpm","forward.xpm","home.xpm" ] ]
        else:
            home = "index.html"

        help = HelpViewer(home, ".", ".history", ".bookmarks", None, "help viewer",icons)
 
        if QApplication.desktop().width() > 400 and QApplication.desktop().height() > 500:
            help.show()
        else:
            help.showMaximized()

        QObject.connect(app, SIGNAL("lastWindowClosed()"),app, SLOT("quit()"))
        app.exec_loop()

    main(sys.argv)

# END
