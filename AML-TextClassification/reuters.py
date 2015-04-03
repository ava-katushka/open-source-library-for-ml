#!/usr/bin/python 
# -*- coding: utf-8 -*- 

from HTMLParser import HTMLParser
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
import re
import sys
import os.path
import numpy as np

"""
Класс-парсер для работы с выборкой Reuters-21578

"""

class ReutersParser(HTMLParser):

    def __init__(self, path_to_reuters, encoding = 'latin-1', sample = "modapte"):
        HTMLParser.__init__(self)
        self.__docs = { "train": [], "test": [] }
       
        #считываем данные о возможных значениях каждой категории
        self.numbers_to_types = {}
        self.all_types_to_numbers = {}
        for filename in glob(os.path.join(data_path, "*.lc.txt")):
            with open(filename, "r") as f:
                categ_name = filename[filename.index("-") + 1 : filename.rindex("-")]
                self.numbers_to_types[categ_name] = [item.strip() for item in f.read().split("\n")]
                self.all_types_to_numbers[ categ_name ] = {}
                for i in range( len(self.numbers_to_types[categ_name]) ):
                    self.all_types_to_numbers[ categ_name ][ self.numbers_to_types[categ_name][i] ] = i
        self.__reset()
        self.__func_sample = "use_" + sample + "_sample"
        self.encoding = encoding

    def use_modapte_sample(self, attrs):
        if attrs[0][1] == "YES":
            if attrs[1][1] == "TRAIN":
                self.is_train_set = 1
                self.is_test_set = 0
            elif attrs[1][1] == "TEST":
                self.is_test_set = 1
                self.is_train_set = 0
        self.__is_document_used = self.is_test_set or self.is_train_set 
        return self.__is_document_used

    def handle_starttag(self, tag, attrs):  
        if ( tag == "reuters"):
            getattr(self, self.__func_sample)(attrs)
        elif (self.__is_document_used):
            if (tag == 'd'):
                self.__is_d_tag = 1
            else:
                 self.__current_tag = tag

    def handle_endtag(self, tag):
        if (self.__is_document_used):
            if ( tag == "reuters" ):
                self.__tags_store[ "body" ] = re.sub(r'\s+', r' ', self.__tags_store[ "body" ])
                if (self.is_train_set):
                    self.__docs["train"].append(self.__tags_store)
                else:
                    self.__docs["test"].append(self.__tags_store)
                self.__reset()
            elif ( tag == 'topics' or tag == 'places' or tag == 'people' or tag == 'orgs' \
                or tag =='exchanges' ): #TODO: or tag == 'companies' нет информации о всех возможных именах компаний

                self.__tags_store[tag] = [ self.all_types_to_numbers[tag][item] for item in self.__d_store]
                self.__is_d_tag = 0
                self.__d_store = []


    def handle_data( self, data ):
        if (self.__is_document_used):
            if self.__is_d_tag:
                self.__d_store.append(data)
                
            elif ( self.__current_tag == "title" or self.__current_tag == "body" ):
                self.__tags_store[ self.__current_tag ] += data

    def __reset(self):
        self.__tags_store = { "topics": '', "places": [], "people": [], "orgs": [], "exchanges": [],\
                    "companies": [], "title": "", "body": "" }
        self.__is_d_tag = 0
        self.__d_store = []
        self.__current_tag = ""
        self.__is_document_used = 0
        self.is_test_set = 0
        self.is_train_set = 0


    def parse(self, files):
        data = files.read()
        self.feed(data)
        #print self.__docs["train"][0]

    """
    Возвращает информацию о документах в следующем формате
    docs[kind_of_sample] - массив документов, где kind_of_sample: "train", "test" - обучающая или тестовая выборка
    Каждый документ - словарь со следующими полями:
    "topics": ''
    "places": []
    "people": []
    "orgs": []
    "exchanges": []
    "companies": []
    "title": ""
    "body": ""


    """
    def get_corpus(self):
        return self.__docs

"""
При запуске программы указывать путь к папке с коллекцией документов Reuters-21578
"""

    
data_path = sys.argv[1]
rp = ReutersParser(data_path)
for filename in glob(os.path.join(data_path, "*.sgm")):
    with open(filename, "r") as f:
        rp.parse(f)
    break


"""
Получаем разреженную матрицу текстовых признаков
"""
count_vect = CountVectorizer()
text_data = [line["body"] for line in rp.get_corpus()["train"]]
matrix = count_vect.fit_transform(text_data)

"""
Простой "наивный" баейсовский классификатор
"""
Y_train = []
for line in rp.get_corpus()["train"]:
    if (len(line["places"]) != 0):
        Y_train.append(line["places"][0])
    else:
        Y_train.append(-1)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(matrix, Y_train)
print np.mean(cross_val_score(clf, matrix, np.array(Y_train)))

