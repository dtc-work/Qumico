import os
import shutil

from os import path

"""
tools/rm_output.py
===
folder_path = './Qumico/out_c'
folder_path = './Qumico/onnx'
===
folder_path = './Qumico/samples/conv/tensorflow/model'
folder_path = './Qumico/samples/conv/tensorflow/onnx'
folder_path = './Qumico/samples/conv/tensorflow/out_c'
===
folder_path = './Qumico/samples/mlp/tensorflow/model'
folder_path = './Qumico/samples/mlp/tensorflow/onnx'
folder_path = './Qumico/samples/mlp/tensorflow/out_c'
===
folder_path = './Qumico/samples/tiny_yolo_v2/tensorflow/out_c'
===
folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/onnx'
folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/out_c'
folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/out_c_optimize'
folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/output'
===
folder_path = './Qumico/samples/vgg16/keras/model'
folder_path = './Qumico/samples/vgg16/keras/onnx'
folder_path = './Qumico/samples/vgg16/keras/out_c'
"""

#get upper_path = /home/qumico/
dir_path = path.abspath(os.pardir)

def rm_base(target):
    """
    get rm_path
        get a rm_dir path from dir_path and os.path.join
    if...else...
        check if there is a folder that you want to REMOVE.
        REMOVE the folder and display "success REMOVE" if it does exist,
        display "do not exist" message if it does not exist.
    """
    if path.exists(target):
        shutil.rmtree(target)
        print("success REMOVE ", target) 
    else:
        print("do not exist ", target)


def rm_outc():
    #folder_path = './Qumico/out_c'
    rm_base(os.path.join(dir_path, 'out_c'))
    #folder_path = './Qumico/onnx'
    rm_base(path.join(dir_path, 'onnx'))


def rm_conv_tensorflow():
    #folder_path = './Qumico/samples/conv/tensorflow/model'
    rm_base(path.join(dir_path, 'samples', 'conv', 'tensorflow', 'model'))
    
    #folder_path = './Qumico/samples/conv/tensorflow/onnx'
    rm_base(path.join(dir_path, 'samples', 'conv', 'tensorflow', 'onnx'))

    #folder_path = './Qumico/samples/conv/tensorflow/out_c'
    rm_base(path.join(dir_path, 'samples', 'conv', 'tensorflow', 'out_c'))


def rm_mlp_tensorflow():
    #folder_path = './Qumico/samples/mlp/tensorflow/model'
    rm_base(path.join(dir_path, 'samples', 'mlp', 'tensorflow', 'model'))

    #folder_path = './Qumico/samples/mlp/tensorflow/onnx'
    rm_base(path.join(dir_path, 'samples', 'mlp', 'tensorflow', 'onnx'))
 
    #folder_path = './Qumico/samples/mlp/tensorflow/out_c'
    rm_base(path.join(dir_path, 'samples', 'mlp', 'tensorflow', 'out_c'))


def rm_tinyyolov2_tensorflow():
    #folder_path = './Qumico/samples/tiny_yolo_v2/tensorflow/out_c'
    rm_base(path.join(dir_path, 'samples', 'tiny_yolo_v2', 'tensorflow', 'out_c'))


def rm_tinyyolov2_yad2k_keras():
    #folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/onnx'
    rm_base(path.join(dir_path, 'samples', 'tiny_yolo_v2_yad2k', 'keras', 'onnx'))

    #folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/out_c'
    rm_base(path.join(dir_path, 'samples', 'tiny_yolo_v2_yad2k', 'keras', 'out_c'))

    #folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/out_c_optimize'
    rm_base(path.join(dir_path, 'samples', 'tiny_yolo_v2_yad2k', 'keras', 'out_c_optimize'))

    #folder_path = './Qumico/samples/tiny_yolo_v2_yad2k/keras/output'
    rm_base(path.join(dir_path, 'samples', 'tiny_yolo_v2_yad2k', 'keras', 'output'))


def rm_vgg16_keras():
    #folder_path = './Qumico/samples/vgg16/keras/model'
    rm_base(path.join(dir_path, 'samples', 'vgg16', 'keras', 'model'))

    #folder_path = './Qumico/samples/vgg16/keras/onnx'
    rm_base(path.join(dir_path, 'samples', 'vgg16', 'keras', 'onnx'))

    #folder_path = './Qumico/samples/vgg16/keras/out_c'
    rm_base(path.join(dir_path, 'samples', 'vgg16', 'keras', 'out_c'))


if __name__ == '__main__':
    
    print("\nQumico's Output REMOVE program\n")

    #rm_temp_outc()
    rm_outc()
    rm_conv_tensorflow()
    rm_mlp_tensorflow()
    rm_tinyyolov2_tensorflow()
    rm_tinyyolov2_yad2k_keras()
    rm_vgg16_keras()
    
    print("\nREMOVE Complete")

