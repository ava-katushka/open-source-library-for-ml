import multiprocessing
import bz2
import gensim
def process_article((title, text, number)):
    text = gensim.corpora.wikicorpus.filter_wiki(text)
    return title.encode('utf8'), gensim.utils.simple_preprocess(text)

def convert_wiki(infile, processes=multiprocessing.cpu_count()):
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes)
    #    texts = gensim.corpora.wikicorpus._extract_pages(bz2.BZ2File(infile)) # generator
        texts = gensim.corpora.wikicorpus._extract_pages(infile) # generator
        ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would try to load the entire dump into RAM...
        for group in gensim.utils.chunkize(texts, chunksize=10 * processes):
            for title, tokens in pool.imap(process_article, group):
                if len(tokens) >= 50 and not any(title.startswith(ignore + ':') for ignore in ignore_namespaces):
                    yield title.replace('\t', ' '), tokens
        pool.terminate()

for title, tokens in convert_wiki('wiki2.xml'):
    print "%s\t%s" % (title, ' '.join(tokens))
