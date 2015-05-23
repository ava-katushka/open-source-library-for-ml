pathCommon = "C:\\Users\\Toshik\\AML\\WikiEntities"

pathTypes = "C:\\Users\\Toshik\\AML\\Entities.txt"
dataTypes = open(pathTypes, 'r')

dataTypesText = dataTypes.read().split('\n')

types = []
for typeStr in dataTypesText:
    type1 = open(pathCommon + "\\Wiki" + typeStr + ".txt", 'r')
    types.append(set(type1.read().split('\n')))

f = open("C:\\Users\\Toshik\\AML\\filename.txt", 'r')
wikiPair = f.read().split(',')

wikiPair[0] = wikiPair[0][3:]

answer = []
for typeStr in dataTypesText:
    answer.append(open("C:\\Users\\Toshik\\AML\\" + "\\Wiki" + typeStr + ".txt", 'w'))

for ent in wikiPair:
    if ent.count('\t') == 0:
        continue
    x = ent[:ent.index('\t')]
    for i in range(0, len(types)):
        if x.count('http://en.wikipedia.org/wiki/') == 0:
            continue
        if x[len('http://en.wikipedia.org/wiki/') + 1:] in types[i]:
            answer[i].write(x[len('http://en.wikipedia.org/wiki/') + 1:] + '\n');
