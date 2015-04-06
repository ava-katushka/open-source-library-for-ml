path = "C:\\Users\\Toshik\\AML\\instance_types_en.nt"
datafile = open(path, 'r')

pathCommon = "C:\\Users\\Toshik\\AML\\Entities"
nameCommon = "<http://dbpedia.org/ontology/"

pathTypes = "C:\\Users\\Toshik\\AML\\Entities.txt"
dataTypes = open(pathTypes, 'r')

dataTypesText = dataTypes.read().split('\n')

types = []
for type in dataTypesText:
    types.append(open(pathCommon + "\\" + type + ".txt", 'w'))

data = datafile.readline()
while (data.__len__() != 0):
    dataset = data.split(' ')
    for i in  range(0, len(dataTypesText)):
        if dataset[2] == nameCommon + dataTypesText[i] + ">":
            types[i].write(dataset[0][len('<http://dbpedia.org/resource/'):len(dataset[0]) - 1] + "\n")

    data = datafile.readline()

for type in types:
    type.close()
