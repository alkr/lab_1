import keras
from sys import argv
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

for k in range(len(train_images)):
	print(train_labels[k])
	h, w = train_images[k].shape
	for i in range(h):
		for j in range(w):
			print(train_images[k][i][j], end = ' ')
		print('')

for k in range(len(test_images)):
	print(test_labels[k])
	h, w = test_images[k].shape
	for i in range(h):
		for j in range(w):
			print(test_images[k][i][j], end = ' ')
		print('')
			
		




