import keras
from keras import backend as K
from qumico.Qumico import Qumico

# qumico インスタンス作成
convert = Qumico()

# backend を推論モードに設定
K.set_learning_phase(0)

# model_type: default is 0
# 0 : download from keras model zoo vgg16 imagenet version
# 1 : retrain by vgg16_train.py, saved json and hdf5 file.
# 2 : retrain by vgg16_train.py, saved yaml and hdf5 file.
model_type = 1

if model_type == 0:
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, input_tensor=None, input_shape=(224, 224, 3))
    convert.conv_keras_to_onnx(model_type='keras', output_path='onnx', model_name='vgg16', cache_path='model',
                               output_op_name=model.output.op.name, K=K)
elif model_type == 1:
    convert.conv_keras_to_onnx(model_type='json', output_path='onnx', model_name='vgg16', cache_path='model',
                               json_file='model/sample.json', h5_file='model/sample.hdf5')
elif model_type == 2:
    convert.conv_keras_to_onnx(model_type='yaml', output_path='onnx', model_name='vgg16', cache_path='model',
                               yaml_file='model/sample.yaml', h5_file='model/sample.hdf5')
else:
    pass
