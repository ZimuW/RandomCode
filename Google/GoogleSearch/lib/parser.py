import re

from bs4 import BeautilfulSoup
import urllib2

class Parser(object):
    def __init__(self, url):
        self.url = url
        self.soup = self.request(self.url)
        self.result = {}

    def request(self, url):
        if url != "":
            request = urllib2.Request(url, None,
                                      {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_4) \
                                        AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11'})
            urlfile = urllib2.urlopen(request)
            page = urlfile.read()
            soup = BeautifulSoup(page, "html.parser")
            return soup

class GoogleParser(Parser):
    def __init__(self, keyword):
        super(GoogleParser, self).__init__("")
        self.keyword = keyword.replace(" ", "%20")
        self.result = []

    def search(self):
        self.url = "https://www.google.com/search?q=%s" % self.keyword
        self.parse(self.url)
        return self.result

    def parse(self, url):
        soup = self.request(url)
        for h3 in soup.findaAll('h3', {"class": "r"}):
            if h3.a.has_attr('href'):
                self.result.append((h3,a['href'], h3.a.string))
        return

class HTMLParser(Parser):
    def parse(self):
        isdocument = self.soup.body.findAll(text=re.compile(r'documentation', re.IGNORECASE))
        if isdocument:
            return self.parseDocumentation()
        isSO = ("stackoverflow.com/question" in self.url and "tagged" not in self.url)
        if isSO:
            return self.parseStackOverflow()
        return None

    def parseDocumentation(self):
        content = []
        self.result['title'] = "Documentation"
        section = self.soup.findAll('div', {"class": "section"})
        if len(section) > 0:
            content.append( unicode(BeautifulSoup(str(section[0]).decode('UTF-8')[:1000])) + "...."
            self.result['content'] = content
        return self.result

    def parseStackOverflow(self):
        content = []
        title = self.soup.find('title').getText()
        self.result['title'] = title
        for posttext in self.soup.findAll('div', {"class": "post-text"}):
            content.append(re.sub('<pre>', '<pre class="prettyprint">', str(posttext).decode('UTF-8')))
        self.result['content'] = content
        return self.result
