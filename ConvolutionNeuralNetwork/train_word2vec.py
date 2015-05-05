# -*- coding: utf-8 -*-
 
import logging
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train_model(inp, outp1, outp2):
    """
    Обучает модель word2vec.
    NOTE: требует больших ресурсов памяти, обучается ОЧЕНЬ долго.
    (у меня ушло ~30ч)
    Можно обучать по кускам, что, вероятно, разумнее.
    :param inp: путь к соханённой с помощью WikiCorpus базы текстов
    :param outp1: путь к модели в бинарном виде
    :param outp2: путь к модели в тектовой виде
    :return:
    """
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    # если в дальнейшем не планируется дообучать модель, то применяем эту функцию
    # с помощью неё удаляется вся лишняя информация и модель занимает гораздо меньше места
    # model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
    # протестируем полученную модель:
    print model.similarity('woman', 'man')


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.INFO)

    # check and process input arguments
    if len(sys.argv) < 4:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    train_model(inp, outp1, outp2)

if __name__ == '__main__':
    main()