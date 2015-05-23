# -*- coding: utf-8 -*-

import urllib
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pathTypes', default = os.getcwd() + '\\Entities.txt')
parser.add_argument('--pathEntities', default = os.getcwd())
parser.add_argument('--pathWikiEntities', default = os.getcwd())
parser.add_argument('--pathInstanceTypes', default = os.getcwd() + '\\instance_types_en.nt')
parser.add_argument('--pathLinks', default = os.getcwd() + '\\wikipedia_links_en.nt')
paths = parser.parse_args(sys.argv[1:])

dataTypes = open(paths.pathTypes, 'r')

dataTypesText = dataTypes.read().split('\n')

types = []
for typeStr in dataTypesText:
    type1 = open(paths.pathEntities + "\\" + typeStr + ".txt", 'r')
    types.append(type1.read().split('\n'))

for j in range(len(types)):
    for i in range(len(types[j])):
        types[j][i] = urllib.unquote(types[j][i])
        types[j][i] = urllib.quote(types[j][i])
    types[j] = set(types[j])

nameCommon = "<http://dbpedia.org/resource/"

Wikitypes = []
for typeStr in dataTypesText:
    Wikitypes.append(open(paths.pathWikiEntities + "\\Wiki" + typeStr + ".txt", 'w'))

datafile = open(paths.pathLinks, 'r')

count = 0

data = datafile.readline()
while (len(data) != 0):
    dataset = data.split(' ')
    #dataset[2] = urllib.unquote(dataset[2]).decode('utf8')
    for i in  range(0, len(dataTypesText)):
        if dataset[2].startswith('<http://dbpedia.org/resource/'):
            if dataset[2][len('<http://dbpedia.org/resource/'):len(dataset[2]) - 1] in types[i]:
                Wikitypes[i].write(urllib.unquote(dataset[0][len('<http://en.wikipedia.org/wiki/'):len(dataset[0]) - 1]) + "\n")

    data = datafile.readline()

for type in Wikitypes:
    type.close()
