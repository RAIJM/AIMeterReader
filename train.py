import tensoflow as tf
from model import MeterModel




batch_size = 50
learning_rate = 0.001
num_epochs =100000
drop_rate = 0.5




def vector_to_one_hot(labels):
	return (np.arange(3) == train_labels[:,None]).astype(np.float32)

def train(path_to_dataset):

	train_labels = []
	valid_labels = []
	test_labels = []
	train_data = []
	valid_data=[]
	test_data=[]

	try:
		with open(pickle_file,'rb') as f:
			save = pickle.load(f)
			train_data = save['train_dataset']
			train_labels = save['train_labels']
			valid_data = save['valid_dataset']
			valid_labels = save['valid_label']
			test_data = save['test_dataset']
			test_labels = save['test_labels']
			del save
	except Exception:
		pass

	num_train_examples = len(train_labels)

	train_labels = vector_to_one_hot(train_labels)
	valid_labels = vector_to_one_hot(valid_labels)
	test_labels = vector_to_one_hot(test_labels)

	x = tf.placeholder(tf.float32,[None,96,96,1])
	y_ = tf.placeholder(tf.float32,[None,3])

	output = Model.inference(x,0.5)
	loss = Model.loss(output,y_)

	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	correct_predicition =tf.equal(tf.argmax(y_,1),tf.argmax(net_output,1))

	accuracy = tf.reduce_mean(tf.cast(correct_predicition,"float32"))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initalizer())

		for epoch_i in range(num_epochs):
			batch_pos = 0
			for i in range(num_train_examples // batch_size):
				batch_xs = train_data[batch_pos:batch_pos+batch_size]
				batch_ys = train_labels[batch_pos:batch_pos+batch_size]
				sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
				batch_pos+=50

			if epoch_i % 100 == 0:
				valid_accuracy = sess.run(accuracy,feed_dict={x:valid_data,y_:valid_labels})
				print("Validation Accuracy ",accuracy)

		test_accuracy = sess.run(accuracy,feed_dict={x:test_data,y_:test_labels})
		print("Test accuracy ", test_accuracy)













