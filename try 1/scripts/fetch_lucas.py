import urllib.request
import re
html = urllib.request.urlopen('https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/').read().decode('utf-8')
links = re.findall(r'href="([^"]+)"', html)
print(links)
