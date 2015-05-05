# -*- coding: utf-8 -*-

import logging
import os.path
import sys
from gensim.corpora import WikiCorpus


def make_wiki_corpus(inp, outp, logger):
    '''
    Предобработка википедии.
    :param inp: путь к файлу, например: enwiki-20150304-pages-articles.xml.bz2
    :param outp: выходной текстовый файл с предобработанной базой текстов
                 например: wiki.en.text
    :param logger: логер для вывода информации о процессе предобработки
    '''
    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})

    i = 0
    space = " "
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i += 1
        if i % 10000 == 0:
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


def main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

        # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    make_wiki_corpus(inp, outp, logger)

if __name__ == '__main__':
    main()
