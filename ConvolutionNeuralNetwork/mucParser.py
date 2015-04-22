def openFile( path ):
    datasetFile = open( path, 'r' )
    return datasetFile

def closeFile ( datasetFile ):
    datasetFile.close()

keysFile = openFile( 'key-tst1-help' )
articlesFile = openFile( 'tst1-muc3-help' )

def readFile ( datasetFile ):
    datasetText = datasetFile.read().split('\n')
    datasetText = [ line.strip() for line in datasetText if len(line) > 0 ]
    return datasetText

keysFileData = readFile( keysFile )
articlesFileData = readFile( articlesFile )
articlesFileData.append('TST1-MUC3')

def loadArticle ( articlesFileData, numStrArticles ):
    for n in range( numStrArticles, len ( articlesFileData ) ):
        if ( articlesFileData[n].find('TST1-MUC3') != -1 ):
            article = ''
            for k in range( numStrArticles, n ):
                article += articlesFileData[k] + ' '
            return n + 1, article

def loadKey ( keysFileData, numStrKeys, numArticle ):

    startKey = numStrKeys;
    keys =[]
    for n in range( numStrKeys, len ( keysFileData ) ):
        if ( keysFileData[n].find('0.  MESSAGE: ID') != -1 ):
            if ( numArticle == int( keysFileData[n][-4:] ) ):
                startKey = n
            else :
                return endKey + 1, keys

            numKey = int( keysFileData[n][-4:] )
        if ( keysFileData[n].find('24. HUM TGT: TOTAL NUMBER') != -1 ):
            endKey = n
            keys.append( keysFileData[ startKey : endKey + 1] )
    return len ( keysFileData ), keys

numStrArticles = 1
numStrKeys = 0
numArticle = 0

def loadLocation( keys ):
    locationStr = []
    for key in keys:
        for s in key:
            if (s.find('3.  INCIDENT: LOCATION') != -1):
                str = '3.  INCIDENT: LOCATION'
                s = s[len('3.  INCIDENT: LOCATION') :]
                s = s.strip()
                locationStr.append(s)

    locationInArticle = []
    classLocationInArticle = []
    for loc in locationStr:
        loc = loc.replace('-', ':')
        loc = loc.split(':')
        loc = [line.strip() for line in loc]
        locHelp = loc
        loc = [line[0: line.find('(')] for line in loc]
        classLoc = [line[line.find('(') + 1: line.find(')')] if line.find('(') != -1 else \
                     'COUNTRY' for line in locHelp]
        for i in range (len(loc)):
            loc[i] += ':' + classLoc[i]
        locationInArticle += loc
        locationInArticle = list(set(locationInArticle))

    return locationInArticle

def searchInArticle ( essentia, article ):
    koord = []
    offset = 0
    essentia = essentia.decode('utf-8')
    while (article.find(essentia) != -1):
        koord.append(offset + article.find(essentia))
        offset += article.find(essentia)
        article = article[article.find(essentia) + len(essentia):]
    return koord



while ( numStrArticles < len( articlesFileData ) ) and ( numStrKeys < len( keysFileData ) ):
    numArticle += 1
    numStrArticles, article = loadArticle( articlesFileData, numStrArticles )
    article = article.decode('utf-8')
    numStrKeys, keys = loadKey( keysFileData, numStrKeys, numArticle )
    #location
    locationInArticle = loadLocation( keys )
    for i in range(len(locationInArticle)):
        koord = searchInArticle( locationInArticle[i][:locationInArticle[i].find(':')], article )
        for j in range(len(koord)):
            print locationInArticle[i][:locationInArticle[i].find(':')],locationInArticle[i][locationInArticle[i].find(':') + 1:] , koord[j], koord[j] + len(locationInArticle)







