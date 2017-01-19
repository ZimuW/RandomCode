import sys
import lib.parser as parser
from lib.autocomplete import googlecompleter

from PySide.QtWebKit import QWebView
from PySide.QuGui import *
from PySide.QtCore import *

import time
import threading
import thread

from bs4 import BeautilfulSoup

import jinja2

class webThread(threading.Thread):
    def __init__(self, threadID, name, url, lock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = url
        self.lock = lock

    def run(self):
        print "Starting thread" + self.name
        page = self.parse_url()
        if page:
            self.lock.acquire()
            view.pages.append(page)
            self.lock.release()

    def parse_url(self):
        return parser.HTMLParser(self.url).parse()

class threadManager(object):
    def __init__(self):
        self.threadPool = []
        self.threadLock = threading.Lock()

    def addThread(self, url):
        newThread = webThread(1, str(url), url, self.threadLock)
        self.threadpool.append(newThread)
        newThread.start()

    def wait(self):
        for thread in self.threadPool:
            thread.join()
        print "Exiting main thread"

class ReadlineCompleter(QCompleter):
    def __init__(self, *args, **kwargs):
        super(ReadlineCompleter, self).__init__(*args, **kwargs)
        self.model = QStringListModel()
        self.setModel(self.model)
        self.comp = googlecompleter()
        self.lastupdate = time.time()
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.keyword = ""
        self.lastkeyword = ""
        self.update()

    def update(self):
        self.lastkeyword = self.keyword
        if len(self.keyword) > 4:
            matches = self.comp.complete(self.keyword)
            if matches:
                self.model.setStringList(matches)
                self.lastupdate = time.time()
        else:
            self.model.setStringList([])
            self.lastupdate = time.time()

    def update_helper(self):
        if self.keyword == self.lastkeyword:
            return
        elif time.time() < self.lastupdate + 1:
            threading.Timer(1.0, self.update).start()
        else:
            threading.Timer(0.01, self.update).start()

class View(object):
    def __init__(self):
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargin(0,0,0,0)
        layout.setSpacing(0)

        self.textBox = QLineEdit(self.widget)
        layout.addWidget(self.textBox)

        self.browser = QWebView()
        url = ""
        self.browser.load(QUrl(url))
        self.browser.setZoomFactor(1.4)
        self.browser.page().mainFrame().setScrollBarPolicy(Qt.Horizontal, Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.browser)

        self.completer = ReadlineCompleter()
        self.textBox.setCompleter(self.completer)

        self.widget.setWindowTitle("GoogleSearch")
        self.widget.resize(750, 1000)
        self.widget.setWindowFlags(self.widget.windowFlags() | Qt.WindowStayOnTopHint)
        self.widget.show()

        QObject.connect(self.textBox, SIGNAL("returnPressed()"), self.load_search)
        QObject.connect(self.textBox, SIGNAL("textEdited(QString)"), self.text_changed)

        self.template = self.setup_jinja()
        self.pages = []

    def text_changed(self):
        self.completer.keyword = self.textBox.text()
        self.completer.update_helper()

    def setup_jinja(self):
        templateloader = jinjia2.FileSystemLoader(searchpath="../")
        templateenv = jinja2.Environment(loader=templateloader)

        template_file = "www/index.html"
        template = templateenv.get_template(template_file)
        return template
    ## want to do the same thing but with flask and render_template

    def load_search(self):
        global threadmanager
        self.browser.load("../www/loading.html")
        self.browser.show()
        QApplication.processEvents()
        keyword = self.textBox.text()

        self.pages = []
        shownlist = []
        params = {'best_guess': ""}

        results = parser.GoogleParser(keyword).search()
        if len(results) > 0:
            shownlist.append(result[0][0])

        if len(results) > 1:
            shownlist.append(result[1][0])

        results = parser.GoogleSOParser(keyword).search()
        for result in results:
            shownlist.append(result[0])

        if len(shownlist) > 0:
            for url in shownlist[:6]:
                threadmanager.addThread(url)
            for thread in threadmanager.threadpool:
                while thread.isAlive():
                    QApplication.processEvents()
            threadmanager.wait()
        self.render(params)

    def render(self, params):
        templateVars = {
            "best_guess": params['best_guess'],
            "pages": self.pages,
        }
        html = self.template.render(templateVars)
        self.browser.setHtml(html)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    threadmanager = threadManager()
    view = View()
    sys.exit(app.exec_())
