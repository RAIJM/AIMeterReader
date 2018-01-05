import tensorflow as tf
from models import MeterModel
import numpy as np
from six.moves import cPickle as pickle




batch_size = 50
learning_rate = 0.0001
num_epochs = 30
drop_rate = 0.5
log_dir = '/tmp/meter'




def vector_to_one_hot(labels):
	labels = np.asarray(labels)
	b = (np.arange(2) == labels[:,None]).astype(np.float32)
	return b

def train(path_to_dataset):

	train_labels = []
	valid_labels = []
	test_labels = []
	train_data = []
	valid_data=[]
	test_data=[]

	try:
		with open(path_to_dataset,'rb') as f:
			save = pickle.load(f)
			train_data = save['train_dataset']
			train_labels = save['train_labels']
			valid_data = save['valid_dataset']
			valid_labels = save['valid_labels']
			test_data = save['test_dataset']
			test_labels = save['test_labels']
			del save
	except Exception as e:
		print(e)

	num_train_examples = len(train_labels)

	print(type(train_labels[0]))

	train_labels = vector_to_one_hot(train_labels)
	valid_labels = vector_to_one_hot(valid_labels)
	test_labels = vector_to_one_hot(test_labels)

	x = tf.placeholder(tf.float32,[None,96,96])
	y_ = tf.placeholder(tf.float32,[None,2])

	x_tensor = tf.reshape(x,[-1, 96, 96, 1])

	output = MeterModel.inference(x_tensor,0.5)
	
	loss = MeterModel.loss(output,y_)

	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	correct_predicition =tf.equal(tf.argmax(y_,1),tf.argmax(output,1))

	accuracy = tf.reduce_mean(tf.cast(correct_predicition,tf.float32))

	# create a summary for our cost and accuracy
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("accuracy", accuracy)

	summary_op = tf.summary.merge_all()

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

		tf.train.write_graph(sess.graph_def, '.', 'meter_model.pbtxt')  

		l = [n.name for n in tf.get_default_graph().as_graph_def().node]
		print(l)

		batch_count = num_train_examples // batch_size
		for epoch_i in range(num_epochs):
			batch_pos = 0
			for i in range(batch_count):
				batch_xs = train_data[batch_pos:batch_pos+batch_size]
				batch_ys = train_labels[batch_pos:batch_pos+batch_size]
				_,loss_val,summary = sess.run([train_step, loss, summary_op],feed_dict={x:batch_xs,y_:batch_ys})
				batch_pos+=batch_size
				print('loss',loss_val)

				writer.add_summary(summary, epoch_i * batch_count + i)


			if epoch_i % 2 == 0:
				valid_accuracy = sess.run(accuracy,feed_dict={x:valid_data,y_:valid_labels})
				print("Epoch: %d Validation Accuracy : %.2f " %(epoch_i,valid_accuracy))

		test_accuracy = sess.run(accuracy,feed_dict={x:test_data,y_:test_labels})
		print("Test accuracy ", test_accuracy)
		save_path = saver.save(sess, "/home/stone/meter_model.ckpt")
  		print("Model saved in file: %s" % save_path)



def main(_):
	train('meterData.pickle')

if __name__ == '__main__':
    tf.app.run(main=main)













