from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import SGD
import os

from samples.vgg16.keras import vgg16_generate_data
import onnx
import keras2onnx
import qumico

# Parameters
classes_num = 5
batch_size = 64
epochs = 20
classes = vgg16_generate_data.extract_classes
all_data_num = 572
x_data = None
y_data = None
test_flag = False
root = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(root, 'train_data', 'flowers')
model = None

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
)

train_generator = None

try:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
except FileNotFoundError:
    print("trainデータがみつかりません。vgg16_generate_data.pyを実行してください。")
    exit(1)

def data_config(_classes_num, _batch_size, _epochs, _classes, _all_data_num, _train_dir, _x_data, _y_data, _test_flag):
    global classes_num
    global batch_size
    global epochs
    global classes
    global all_data_num
    global x_data
    global y_data
    global test_flag
    global train_dir

    classes_num = _classes_num
    batch_size = _batch_size
    epochs = _epochs
    classes = _classes
    all_data_num = _all_data_num
    train_dir = _train_dir
    x_data = _x_data
    y_data = _y_data
    test_flag = _test_flag


def train(train_generator):
    # Kerasモデル準備します。
    global model
    model = VGG16(weights='imagenet', include_top=False, input_tensor=None, input_shape=(224,224,3))
     
    
    # 全結合層 FC layer を再構築します。
    x = model.output
    x = GlobalAveragePooling2D(data_format="channels_last")(x)
    x = Dense(1024, activation='relu', kernel_initializer=TruncatedNormal(seed=0))(x)
    
    prediction = Dense(classes_num, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=prediction)
    # VGG16 14layers までを再学習させないよう、固定します。
    for layer in model.layers[:15]:
        layer.trainable = False
    
    
    model.compile(optimizer=SGD(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    model.summary()

    if test_flag:
        hist = model.fit(x=x_data, y=y_data, epochs=epochs, verbose=1)
    else:
        hist = model.fit(train_generator, epochs=epochs, verbose=1)

def main():
    train(train_generator)
    # save keras model
    model_folder = 'model'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    json_file = os.path.join(model_folder, 'sample.json')
    yaml_file = os.path.join(model_folder, 'sample.yaml')
    h5_file = os.path.join(model_folder, 'sample.hdf5')

    json_string = model.to_json()
    open(json_file, 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(yaml_file, 'w').write(yaml_string)
    model.save_weights(h5_file)

    onnx_folder = 'onnx'
    if not os.path.exists(onnx_folder):
        os.mkdir(onnx_folder)
    onnx_path = os.path.join(onnx_folder, 'sample.onnx')
    onnx_model = keras2onnx.convert_keras(model, 'sample', target_opset=qumico.SUPPORT_ONNX_OPSET)
    onnx.save_model(onnx_model, onnx_path)

    print(h5_file, "を作成しました。")
    print('onnxファイルを生成しました。出力先:', onnx_path)


if __name__ == "__main__":
    main()

