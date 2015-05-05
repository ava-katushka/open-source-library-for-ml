import os
from scipy.misc import imread
import sys
import pandas as pd


#Скрипт получает адрес папки и добавляет все картинки в ней в ксв-файл
#Формат картинки 48*48 чисел от 0 до 255 + 1 число обозначающее номер класса (задается в командной строке - все картинки в одной папке принадлежат
#к одному классу
#У нас всего 2 ксв-файла - 0.9 частей выборки уходят в трейновый, остальные в тестовый
def imagesToCsv(train_path='./train_data.csv', test_path = './test_data.csv', path=None, cls=None):
        sz = len(os.listdir(path))
        idx = int(0.9*sz)
        for filename in os.listdir(path)[:idx]:
            if filename!='.DS_Store' and filename[-3:]=='jpg':
                b = imread(path+filename,flatten=0).flatten()
                out = ','.join('%d'%i for i in b)
                out = ','.join([out, str(cls) + '\n'])
                with open(train_path, 'a') as f:
                    f.write(out)
                    f.close()
        for filename in os.listdir(path)[idx:]:
            if filename!='.DS_Store' and filename[-3:]=='jpg':
                b = imread(path+filename,flatten=0).flatten()
                out = ','.join('%d'%i for i in b)
                out = ','.join([out, str(cls) + '\n'])
                with open(test_path, 'a') as f:
                    f.write(out)
                    f.close()

path = sys.argv[1]
cls = sys.argv[2]
imagesToCsv(path=path, cls=cls)
