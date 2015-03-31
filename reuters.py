#!/usr/bin/python 
# -*- coding: utf-8 -*- 

from HTMLParser import HTMLParser
from glob import glob
import re
import sys
import os.path

"""
Класс-парсер для работы с выборкой Reuters-21578

"""

class ReutersParser(HTMLParser):

    def __init__(self, encoding = 'latin-1', sample = "modapte"):
        HTMLParser.__init__(self)
        self.__docs = { "train": [], "test": [] }
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
            #self.__use_modapte_sample( attrs )
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
                or tag =='exchanges' or tag == 'companies'):
                self.__tags_store[tag] = self.__d_store
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

"""
При запуске программы указывать путь к папке с коллекцией документов Reuters-21578
"""
rp = ReutersParser()
data_path = sys.argv[1]
for filename in glob(os.path.join(data_path, "*.sgm")):
    rp.parse(open(filename, "r"))