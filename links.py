pathCommon = "C:\\Users\\Toshik\\AML\\Entities"

pathTypes = "C:\\Users\\Toshik\\AML\\Entities.txt"
dataTypes = open(pathTypes, 'r')

dataTypesText = dataTypes.read().split('\n')

types = []
for typeStr in dataTypesText:
    type1 = open(pathCommon + "\\" + typeStr + ".txt", 'r')
    types.append(set(type1.read().split('\n')))

pathCommon = "C:\\Users\\Toshik\\AML\\WikiEntities"
nameCommon = "<http://dbpedia.org/resource/"

Wikitypes = []
for typeStr in dataTypesText:
    Wikitypes.append(open(pathCommon + "\\Wiki" + typeStr + ".txt", 'w'))

path = "C:\\Users\\Toshik\\AML\\wikipedia_links_en.nt"
datafile = open(path, 'r')

count = 0

data = datafile.readline()
while (data.__len__() != 0):
    dataset = data.split(' ')
    for i in  range(0, len(dataTypesText)):
        if dataset[2].startswith('<http://dbpedia.org/resource/'):
            if dataset[2][len('<http://dbpedia.org/resource/'):len(dataset[2]) - 1] in types[i]:
                Wikitypes[i].write(dataset[0][len('<http://en.wikipedia.org/wiki/'):len(dataset[0]) - 1] + "\n")

    data = datafile.readline()

for type in Wikitypes:
    type.close()
