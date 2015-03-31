path = "C:\\Users\\Toshik\\AML\\wikipedia_links_en.nt"
datafile = open(path, 'r')
link = dict([])
dataall = []

for i in range(1, 1000):
    data = datafile.readline()
    dataset = data.split(' ')
    if (dataset[0].find("http://en.wikipedia.org/wiki/Benjamin_Tucker")!= -1 and dataset[2].find("dbpedia.org") != -1):
        dataall.append(dataset[2])

path = "C:\\Users\\Toshik\\AML\\instance_types_en.nt"

print dataall

datafile = open(path, 'r')
datalen = len(datafile.readlines())
print datalen
data = datafile.readline()
while (data.__len__() != 0):
    dataset = data.split(' ')
    if dataall.count(dataset[0]) != 0:
        print dataset[2]
    data = datafile.readline()