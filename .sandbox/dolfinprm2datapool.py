import sys, re
from scitools.misc import str2obj, str2bool

# NOTE: should first run the .h file through cpp with all the right
# options to filter out all the right branches in #ifdef's!

dolfin_addline_regex = re.compile(r'add\s*\("(.+?)"\s*,\s*(.+)\)\s*;')
dolfin_heading_regex = re.compile(r'//\s*--+\s+(.+)\s+--+')

def read_dolfin_prmfile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    menus = {}
    heading = None
    for line in lines:
        # check for heading (new submenu):
        m = dolfin_heading_regex.search(line)
        if m:
            heading = m.group(1)
            menus[heading] = []
        # check for parameter definition:
        m = dolfin_addline_regex.search(line)
        if m:
            if not heading:
                print 'No heading before "add" call - error'
                sys.exit(1)
            name = m.group(1).strip()
            value = m.group(2).strip()
            if value[0] == '"' and value[-1] == '"':  # C++ string, strip ""
                value = value[1:-1]
            #if value == 'true': value = True
            #if value == 'false': value = False
            # find out if value is float or int or string?
            value = str2obj(value)
            if isinstance(value, float):
                s2t = float
            elif isinstance(value, bool):
                s2t = str2bool
            elif isinstance(value, int):
                s2t = int
            elif isinstance(value, basestring):
                s2t = str
            menus[heading].append((name, value, s2t))
    return menus

import pprint
pprint.pprint(read_dolfin_prmfile(sys.argv[1]))
