import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pathArticles', default = os.getcwd() + '\\WikiArticles')
parser.add_argument('--pathLinks', default = os.getcwd() + '\\WikiLinks')
parser.add_argument('--pathCorpus', default = os.getcwd() + '\\Corpus')
paths = parser.parse_args(sys.argv[1:])
if os.path.exists(paths.pathCorpus) == False:
    os.mkdir(paths.pathCorpus)

if os.path.exists(paths.pathLinks) == False or os.path.exists(paths.pathFrticles) == False:
    print "Bad directories!!!"
else:

    os.chdir(paths.pathCorpus)
    import shutil
    from os import listdir
    x = listdir(paths.pathLinks)
    for q in x[2:]:
        w = listdir(paths.pathLinks + "\\" + q)
        os.mkdir(q)
        for ww in w:
            if ww == 'PaxHeader':
                continue
            os.mkdir(q+'\\'+ww)
            shutil.copy2(paths.pathLinks + "\\" + q + '\\' + ww, paths.pathCorpus + "\\" + q +'\\' + ww+'\\links')
            shutil.copy2(paths.pathArticles + "\\" + q + '\\' + ww, paths.pathCorpus + "\\" + q +'\\' + ww+'\\article')