import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pathTypes', default = os.getcwd() + '\\Entities.txt')
parser.add_argument('--pathEntities', default = os.getcwd() + 'Entities')
parser.add_argument('--pathInstanceTypes', default = os.getcwd() + '\\instance_types_en.nt')
paths = parser.parse_args(sys.argv[1:])

datafile = open(paths.pathInstanceTypes, 'r')

nameCommon = "<http://dbpedia.org/ontology/"

dataTypes = open(paths.pathTypes, 'r')

dataTypesText = dataTypes.read().split('\n')

types = []
for type in dataTypesText:
    types.append(open(paths.pathEntities + "\\" + type + ".txt", 'w'))

data = datafile.readline()
while (len(data) != 0):
    dataset = data.split(' ')
    for i in  range(0, len(dataTypesText)):
        if dataset[2] == nameCommon + dataTypesText[i] + ">":
            types[i].write(dataset[0][len('<http://dbpedia.org/resource/'):len(dataset[0]) - 1] + "\n")

    data = datafile.readline()

for type in types:
    type.close()
