
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def tokenize(file_text):
    tokens = [word for sent in nltk.sent_tokenize(file_text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search(u'[a-zA-Zа-яА-Я]', token):
            filtered_tokens.append(token)
    stop_words = stopwords.words('russian')
    stop_words.extend([u'что', u'это', u'так', u'вот', u'быть', u'как', u'в', u'—', u'к', u'на'])
    tokens = [i for i in filtered_tokens if ( i not in stop_words )]
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(t) for t in tokens]

tokens = map(tokenize, raw_texts)