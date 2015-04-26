#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import os
from HTMLParser import HTMLParser
from glob import glob



"""
Класс-парсер для работы с корпусом Reuters-21578
Реализация предполагает использование выборки modapte

"""

class ReutersParser(HTMLParser):
    """
    self.__docs[kind_of_sample] - массив документов, где kind_of_sample: "train", "test" - обучающая или тестовая выборка
    Каждый документ - словарь со следующими полями:
    "topics": []
    "places": []
    "people": []
    "orgs": []
    "exchanges": []
    "companies": []
    "title": ""
    "body": ""
    """

    def __init__(self, path_to_reuters, encoding = 'latin-1', sample = "modapte"):
        HTMLParser.__init__(self)
        self.__docs = { "train": [], "test": [] }
        self.data_path = path_to_reuters
        #считываем данные о возможных значениях каждой категории
        self.numbers_to_types = {}
        self.all_types_to_numbers = {}
        for filename in glob(os.path.join(path_to_reuters, "*.lc.txt")):
            with open(filename, "r") as f:
                categ_name = filename[filename.index("-") + 1 : filename.rindex("-")]
                self.numbers_to_types[categ_name] = [item.strip() for item in f.read().split("\n")]
                self.all_types_to_numbers[ categ_name ] = {}
                for i in range( len(self.numbers_to_types[categ_name]) ):
                    self.all_types_to_numbers[ categ_name ][ self.numbers_to_types[categ_name][i] ] = i
		    print categ_name
        self.__reset()
        self.__func_sample = "use_" + sample + "_sample"
        #self.encoding = encoding

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
            elif ( tag == "topics" or tag == "places" or tag == "people" or tag == "orgs" \
                or tag =="exchanges" ): #TODO: or tag == 'companies' нет информации о всех возможных именах компаний

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
        self.__tags_store = { "topics": [], "places": [], "people": [], "orgs": [], "exchanges": [],\
                    "companies": [], "title": "", "body": "" }
        self.__is_d_tag = 0
        self.__d_store = []
        self.__current_tag = ""
        self.__is_document_used = 0
        self.is_test_set = 0
        self.is_train_set = 0


    def parse(self):
        for filename in glob(os.path.join(self.data_path, "*.sgm")):
            with open(filename, "r") as f:
                data = f.read()
                self.feed(data)

  
    def get_corpus(self, subset, category, value):
        """
        subset: 'train' or 'test'
            Выбираем подмножество документов корпуса: 'train' - обучающая выборка,
            'test' - тестовая выборка
        category: "topics", "places", "people", "orgs", "exchanges", "companies"
            Выбираем параметр, по которому будем проводить классификацию

        value: "data" or "target"
        """

        if (value == "data"):
            return [line["body"] for line in self.__docs[subset] if len(line[category]) != 0]
        if (value == "target"):
            return [line[category][0] for line in self.__docs[subset] if len(line[category]) != 0]


