import tensorflow as tf
import numpy as np



def get_similarity(word1,word2):
	return 0



def get_score():
	checkpoint_path = "../results_output_gate/1478218222/checkpoint"
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		for v in tf.all_variables():
    		print(v.name)




  if __name__ == '__main__':
	get_score()