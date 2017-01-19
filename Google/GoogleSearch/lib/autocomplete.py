import urllib2
import json

class googlecompleter():
    def complete(self, keyword):
        try:
            keyword = keyword.replace(" ", "%20")
            url = "http://suggestqueries.google.com/complete/search?client=firefox&q=%s" % keyword
            response = urllib2.urlopen(url, timeout=1)
            data = json.load(response)
            return data[1]
        except:
            return None
