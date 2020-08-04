import os
from os import path

import onnx

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

os.environ['TF_KERAS'] = '1'
import keras2onnx

from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

from qumico import SUPPORT_ONNX_OPSET
from samples.text_classification.keras.utils import load_label_encoder_sklearn

def conv_label_encoder_model(encoder, export_onnx_fpath, 
                             target_opset=SUPPORT_ONNX_OPSET,
                             name="scikit-learn label encoder"):
    
    if not path.exists(path.dirname(export_onnx_fpath)):
        os.mkdir(path.dirname(export_onnx_fpath))

    model_onnx = convert_sklearn(encoder,
                             name,
                             [("input", StringTensorType())], # [("input", StringTensorType([None]))],
                            target_opset=target_opset)

    with open(export_onnx_fpath, "wb") as f:
        f.write(model_onnx.SerializeToString())
        

def conv_model_to_onnx(export_onnx_fpath,name, 
                       hdf5_path, json_path,
                       target_opset=SUPPORT_ONNX_OPSET):

    K.clear_session()
    K.set_learning_phase(0)

    if not path.exists(path.dirname(export_onnx_fpath)):
        os.mkdir(path.dirname(export_onnx_fpath))

    model = model_from_json(open(json_path).read())
    model.load_weights(hdf5_path)

    onnx_model = keras2onnx.convert_keras(model, name,
                                          doc_string='', 
                                          target_opset=target_opset,
                                          channel_first_inputs=None)

    onnx.save_model(onnx_model, os.path.join(export_onnx_fpath))


def merge_onnx(): 
    # merge two onnx files of label encoder & model
    pass


if __name__ == "__main__":

    model_path = path.join(path.dirname(__file__),"model")
    # encoder
    # encoder_path = path.join(model_path, "label_encoder.pickle")
    # encoder = load_label_encoder_sklearn(encoder_path)
    # export_fpath = path.join(path.dirname(__file__), "onnx", "LabelEncoder.onnx") 
    # conv_label_encoder_model(encoder, export_fpath)
    
    # kears model
    hdf5_path = path.join(model_path, "TweetDisaster.hdf5")
    json_path = path.join(model_path, "TweetDisaster.json")

    export_onnx_path = path.join(path.dirname(__file__),"onnx", "TweetDisaster.onnx")
    conv_model_to_onnx(export_onnx_path,
                       name="TweetDisaster",
                       hdf5_path=hdf5_path,
                       json_path=json_path,
                       target_opset=SUPPORT_ONNX_OPSET)
    print("ONNXファイルを生成しました。出力先:", export_onnx_path)