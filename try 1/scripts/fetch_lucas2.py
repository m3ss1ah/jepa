import urllib.request
import re
html = urllib.request.urlopen('https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/B11/').read().decode('utf-8')
links = re.findall(r'href="([^"]+)"', html)
print(len(links))
print(links[:20])
