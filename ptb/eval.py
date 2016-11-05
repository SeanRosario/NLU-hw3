import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_embeddings(path_to_location):
	checkpoint = tf.train.latest_checkpoint(path_to_location)
	with tf.Session() as sess:
		saved_model = tf.train.import_meta_graph("{}.meta".format(checkpoint))
		saved_model.restore(sess, checkpoint)
		model = tf.get_default_graph()
		
		global embedding

		embedding = tf.get_default_graph().get_tensor_by_name("Model/embedding:0").eval()



def load_vocabulary():
	global vocabulary
	vocabulary = pickle.load( open( "vocab.p", "rb" ) )
	#print type(vocabulary)
	

def similarity( w1, w2):
	w1 = w1 if w1 in vocabulary else "<unk>"
	w2 = w2 if w2 in vocabulary else "<unk>"
	v1 = embedding[vocabulary[w1]]
	v2 = embedding[vocabulary[w2]]

	#similarity = 1 - distance.cosine(v1, v2)
	similarity = float(cosine_similarity(v1,v2))
	#rint "similarity is ", similarity
	#print v1, v2

	return similarity



def score():
    score = 0
    score += similarity('a', 'an') > similarity('a', 'document')
    score += similarity('in', 'of') > similarity('in', 'picture')
    score += similarity('nation', 'country') >  similarity('nation', 'end')
    score += similarity('films', 'movies') > similarity('films', 'almost')
    score += similarity('workers', 'employees') > similarity('workers', 'movies')
    score += similarity('institutions', 'organizations') > similarity('institution', 'big')
    score += similarity('assets', 'portfolio') > similarity('assets', 'down')
    score += similarity("'", ",") > similarity("'", 'quite')
    score += similarity('finance', 'acquisition') > similarity('finance', 'seems')
    score += similarity('good', 'great') > similarity('good', 'minutes')
    return score


def get_score():
	checkpoint_path = "/Users/SeansMBP/Desktop/Cho/A3/default_results/1478295132/"
	load_embeddings(checkpoint_path)
	load_vocabulary()
	print "The model score is:" ,str(score())



def plot_vectors():
        pca = PCA(n_components=2)
        two_dim = pca.fit_transform(embedding)

        subset_two_dim = two_dim[:300]
        words = [(k, v) for k, v in vocabulary.items()]
        subset_words = words[:300]
        fig, ax = plt.subplots()
        ax.scatter(subset_two_dim[:, 0], subset_two_dim[:, 1])
        for i, word in enumerate(subset_words):
            ax.annotate(word[0], (subset_two_dim[i, 0], subset_two_dim[i, 1]))
        plt.savefig("tSNE_viz.png")




if __name__ == '__main__':
	get_score()
	plot_vectors()
