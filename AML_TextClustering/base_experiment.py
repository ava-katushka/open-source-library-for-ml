from sklearn.feature_extraction.text import TfidfVectorizer
import os 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1,
                                 use_idf=True, tokenizer=tokenize, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(raw_texts) 

print(tfidf_matrix.shape)
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
dist = 1 - cosine_similarity(tfidf_matrix)
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist) 
xs, ys = pos[:, 0], pos[:, 1]

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
groups = df.groupby('label')
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) 
col_dict = dict(zip(range(4), ['red', 'black', 'blue', 'yellow']))
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,  
            mec='none', color=col_dict[name])
    ax.set_aspect('auto')
    ax.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis= 'y', which='both', left='off', top='off',  labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point