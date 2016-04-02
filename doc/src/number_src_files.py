"""
Copy src files with logical names to numbered src files in
published ../../src file directory.
"""

# Grep the sequence of source files from the document
import commands, re, os, sys, shutil, collections
# Look into tmp_preprocess_... for a complete file before mako substitutions
cmd = r"""grep -o -h -E 'prog\[".*?"\]' tmp_preprocess__ftut.do.txt"""
failure, output = commands.getstatusoutput(cmd)
cmd = r"""grep -o -h -E "prog\['.*?'\]" tmp_preprocess__ftut.do.txt"""
failure2, output2 = commands.getstatusoutput(cmd)
output += output2
#print 'output:', output

filenames = []
for line in output.splitlines():
    names = re.findall(r'prog\["(.*?)"\]', line)
    for name in names:
        if name not in filenames:
            filenames. append(name)
counter = 1
prog = collections.OrderedDict()
for filename in filenames:
    new_filename = 'fenics_tutorial_%02d_%s' % (counter, filename)
    print new_filename
    #shutil.copy(os.path.join('src', filename),
    #            os.path.join(os.pardir, os.pardir, 'src', new_filename))
    counter += 1
    prog[filename] = new_filename

# Copy source files

os.chdir('src')
os.system('sh clean.sh')
os.chdir(os.pardir)
# First copy to new names in local scratch directory
if os.path.isdir('tmp'):
    shutil.rmtree('tmp')
shutil.copytree('src', 'tmp')
for dirpath, dirnames, filenames in os.walk('tmp'):
    for filename in filenames:
        print dirpath, filename
        path = os.path.join(dirpath, filename)
        base, ext = os.path.splitext(filename)
        #if len(r) < 2:
        #    base, ext = r
        if base in prog:
            # Rename file
            new_path = os.path.join(dirpath, prog[base] + ext)
            os.rename(path, new_path)

os.system('python rsync_git.py tmp ../../src')

print 'prog = {'
for p in prog:
    print "  '%s': '%s'," % (p, prog[p])
print '}'
# Write prog dictionary into ftut.do.txt
