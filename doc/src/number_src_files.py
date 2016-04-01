"""
Copy src files with logical names to numbered src files in
published src file directory.
"""

# Grep the sequence of source files from the document
import commands, re, os, sys, shutil, collections
cmd = r"""grep -o -h -E 'prog\[".*?"\]' tmp_preprocess__ftut.do.txt"""
failure, output = commands.getstatusoutput(cmd)
filenames = []
for line in output.splitlines():
    names = re.findall(r'prog\["(.*?)"\]', line)
    for name in names:
        if name not in filenames:
            filenames. append(name)
counter = 1
prog = collections.OrderedDict()
for filename in filenames:
    filename += '.py'
    new_filename = 'fenics_tutorial_%02d_%s.py' % (counter, filename)
    print new_filename
    #shutil.copy(os.path.join('src', filename),
    #            os.path.join(os.pardir, os.pardir, 'src', new_filename))
    counter += 1
    prog[filename] = new_filename

print prog
# Write prog dictionary into ftut.do.txt
