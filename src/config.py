# data
dataset = 'test_dataset'
dataset_name = 'test_dataset'
image_shape = (224, 224)

image_width = 224
image_height = 224

learning_rate = 0.001
loss = 'sparse_categorical_crossentropy'
epochs = 7

# cnn & model
input_shape = (224, 224, 3)
output_shape = 1

padding_same = 'same'

activation_relu = 'relu'
activation_sigmoid = 'sigmoid'
activation_softmax = 'softmax'

filters1 = 16
kernel_size1 = 3
pool_size1 = (3, 3)

filters2 = 32
kernel_size2 = 3
pool_size2 = (3, 3)

filters3 = 64
kernel_size3 = 3
pool_size3 = (3, 3)

units = 512

# resnet
resnet_input_shape = (224, 224, 3)
resnet_include_top = True
resnet_weights = 'imagenet'

# logger
histogram_freq = 1
profile_batch = 3