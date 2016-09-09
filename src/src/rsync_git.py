#!/usr/bin/env python
"""
Sync two directory trees with rsync and perform corresponding
git operations (add or rm).
Skip files listed in $HOME/1/.rsyncexclude.

Usage:   rsync_git.py from-dir to-dir
Example: rsync_git.py src-mychap $HOME/repos/pub/mybook/src/mychap

The from-dir is the source and the to-dir is the destination
(e.g. a public directory where resources are exposed).
The script must be run from a dir within the repo of to-dir.
"""

# Typical rsync output:
"""
sending incremental file list
deleting decay7.py
decay_TULL.py

sent 675 bytes  received 34 bytes  1418.00 bytes/sec
total size is 94788  speedup is 133.69
"""

import commands, os, sys

from_ = sys.argv[1]
to_ = sys.argv[2]
cmd = 'rsync -rtDvz -u -e ssh -b --exclude-from=$HOME/1/.rsyncexclude --suffix=.rsync~ --delete --force %s/ %s' % (from_, to_)
print cmd
failure, output = commands.getstatusoutput(cmd)
print output

delete = []
add = []
for line in output.splitlines():
    relevant_line = True
    for text in 'sending incremental file list', \
        'sent ', 'total size is':
        if line.startswith(text):
            relevant_line = False
    if relevant_line and line != '':
        if line.startswith('deleting'):
            delete.append(line.split()[1])
        else:
            add.append(line.strip())

print delete
print add

for filename in delete:
    option = '-rf' if os.path.isdir('%s/%s' % (to_, filename)) else '-f'
    cmd = 'git rm %s %s/%s' % (option, to_, filename)
    print cmd
    os.system(cmd)
for filename in add:
    cmd = 'git add %s/%s' % (to_, filename)
    print cmd
    os.system(cmd)
