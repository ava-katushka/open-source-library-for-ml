import multiprocessing
import bz2
import sys, os
import py_compile
from codecs import open

# py_compile.compile(
#     "/home/daria/anaconda/lib/python2.7/site-packages/gensim-0.10.3-py2.7-linux-x86_64.egg/gensim/corpora/wikicorpus.py")
# py_compile.compile(
#     "/home/daria/anaconda/lib/python2.7/site-packages/gensim-0.10.3-py2.7-linux-x86_64.egg/gensim/utils.py")
#
# import gensim

'''sys.stdout = open("/home/daria/PycharmProjects/first/Anarchism", 'w', 'utf8')'''

import ourwikicorpus
import ourutils
def process_article((title, text, number)):
    text = ourwikicorpus.filter_wiki(text)
  #  print gensim.corpora.wikicorpus.remove_markup(text)
    #print gensim.utils.simple_preprocess(doc=text)
    return title.encode('utf8'), ourutils.simple_preprocess(text)


import re

def convert_wiki(infile, processes=multiprocessing.cpu_count()):
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes)
        texts = ourwikicorpus._extract_pages(bz2.BZ2File(infile)) # generato
        ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft Notes References'.split()
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would try to load the entire dump into RAM...
        for group in ourutils.chunkize(texts, chunksize=10 * processes):
            for title, tokens in pool.imap(process_article, group):
                if len(tokens) >= 50 and not any(title.startswith(ignore + ':') for ignore in ignore_namespaces):
                    yield title.replace('\t', ' '), tokens
        pool.terminate()

i = 0



try:
    os.mkdir("pioNER_Wiki_Articles")
except OSError, e:
    if e.errno != 17:
        raise

    pass
import sys
#Параметр скрипта - путь к архиву с dump Wikipedia, например 'enwiki-20150304-pages-articles.xml.bz2'
for title, tokens in convert_wiki(sys.argv[1]):
    listdir = os.listdir("pioNER_Wiki_Articles/")
    dirName = 'different'
    if ( re.match(('\w+'), title) != None ):
        dirName = title[0]
    if (listdir.count(dirName) == 0):
        os.mkdir("pioNER_Wiki_Articles/" + dirName)
    if (len(os.listdir("pioNER_Wiki_Articles/" + dirName)) == 2500):
        continue
    file = open("pioNER_Wiki_Articles/" + dirName + "/" +title.replace('/', '_'), "w+", "utf8")
    file.write(' '.join(tokens))
    file.close()
    i = i + 1
print "ok"