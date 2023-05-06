import os

mnist_url_base = 'http://yann.lecun.com/exdb/mnist/'
mnist_gz = {
    'X_train': 'train-images-idx3-ubyte.gz',
    'Y_train': 'train-labels-idx1-ubyte.gz',
    'X_test': 't10k-images-idx3-ubyte.gz',
    'Y_test': 't10k-labels-idx1-ubyte.gz'
}
dataset_dir = os.path.join(os.getcwd(), 'dataset')
images_dir = os.path.join(os.getcwd(), 'images')
