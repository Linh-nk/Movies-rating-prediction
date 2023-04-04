from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE

df = pd.read_csv("movies_metadata.csv")
text = df["overview"]

vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                             min_df=1,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
tfidf = vector.fit_transform(text.values.astype('U'))

# Clustering  Kmeans
k = 200
kmeans = MiniBatchKMeans(n_clusters = k)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]

    
request_transform = vector.transform(text.values.astype('U'))
# new column cluster based on the description
df['cluster'] = kmeans.predict(request_transform) 

df.to_csv("movies_metadata_with_cluster_name.csv")