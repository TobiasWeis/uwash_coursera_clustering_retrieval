#!/usr/bin/python
import sframe                            # see below for install instruction
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
#%matplotlib inline

wiki = sframe.SFrame('./data/people_wiki.gl/')
wiki = wiki.add_row_number()             # add row number, starting at 0

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

word_count = load_sparse_csr('data/people_wiki_word_count.npz')
map_index_to_word = sframe.SFrame('data/people_wiki_map_index_to_word.gl/')

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

print wiki[wiki['name'] == 'Barack Obama']

distances,indices = model.kneighbors(word_count[35817], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id','name','distance']]

# why have obama and Francisco Barrio considered close neighbors?
def unpack_dict(matrix, map_index_to_word):
    table = list(map_index_to_word.sort('index')['category'])
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)

def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

obama_words = top_words('Barack Obama')
print obama_words

barrio_words = top_words('Francisco Barrio')
print barrio_words

# use join to combine those two tables
combined_words = obama_words.join(barrio_words, on='word')

# rename the count-column to give explicit names
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})

# obtain most common words by sorting:
common_words = set(combined_words.sort('Obama', ascending=False).topk('Obama', 5)['word'])

#####################################################
# QUIZ:
print "QUIZ: ==================== 5 most frequent words of combined in Obama, how many of the articles in the wikipedia dataset contain all of those 5 words?"
print combined_words

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    if common_words.issubset(set(word_count_vector.keys())):
        return True
    else:
        return False

#####################################################
wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
print "Quiz Question: How many of the articles in the Wikipedia dataset contain all of those 5 words? ==>", wiki['has_top_words'].sum()

#Quiz Question. Measure the pairwise distance between the Wikipedia pages of 
# Barack Obama, George W. Bush, and Joe Biden. 
# Which of the three pairs has the smallest distance?
#
#Hint: For this question, take the row vectors from the word count matrix that correspond to Obama, Bush, and Biden. 
# To compute the Euclidean distance between any two sparse vectors, use sklearn.metrics.pairwise.euclidean_distances.
from sklearn.metrics.pairwise import euclidean_distances # (THIS DOES NOT WORK ON DICTS!)
import math
def get_val(X,idx):
    try:
        return X[X['word'] == idx]['count'][0]
    except:
        try:
            return X[X['word'] == idx]['weight'][0]
        except:
            return 0    
        return 0

def mypairwise_dist(X,Y):
    return math.sqrt(sum((get_val(X,d) - get_val(Y,d))**2 for d in set(X['word']) | set(Y['word'])))

#####################################################    
print "Quiz Question: Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden. Which of the three pairs has the smallest distance? =>",
print "Distance Obama/Biden: ", mypairwise_dist(top_words('Barack Obama'), top_words('Joe Biden'))
print "Distance Bush/Biden: ", mypairwise_dist(top_words('George W. Bush'), top_words('Joe Biden'))
print "Distance Obama/Bush: ", mypairwise_dist(top_words('Barack Obama'), top_words('George W. Bush'))

print "Quiz Question. Collect all words that appear both in Barack Obama and George W. Bush pages. Out of those words, find the 10 words that show up most often in Obama's page."
obama_words = top_words('Barack Obama')
bush_words = top_words('George W. Bush')
combined_words = obama_words.join(bush_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Bush'})
print combined_words.sort('Obama', ascending=False).topk('Obama', 10)['word']

#####################################################
# Extract the TF-IDF vectors
tf_idf = load_sparse_csr('data/people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id', 'name', 'distance']]

def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

obama_tf_idf = top_words_tf_idf('Barack Obama')
print obama_tf_idf

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
print schiliro_tf_idf

# Quiz Question. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?
combined_words = obama_tf_idf.join(schiliro_tf_idf, on='word').rename({'weight':'Obama', 'weight.1':'Schiliro'})
common_words = set(combined_words.sort('Obama', ascending=False).topk('Obama', 5)['word'])
wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
print "Quiz Question. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words? => ",
print wiki['has_top_words'].sum()

###################################

print "Quiz Question. Compute the Euclidean distance between TF-IDF features of Obama and Biden. => ",
print "Distance Obama/Biden: %.3f" % mypairwise_dist(top_words_tf_idf('Barack Obama'), top_words_tf_idf('Joe Biden'))

###################################
# Comptue length of all documents
def compute_length(row):
    return len(row['text'].split(' '))
wiki['length'] = wiki.apply(compute_length)

# Compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
nearest_neighbors_euclidean = wiki.join(neighbors, on='id')[['id', 'name', 'length', 'distance']].sort('distance')
print nearest_neighbors_euclidean

plt.figure(figsize=(10.5,4.5))
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
