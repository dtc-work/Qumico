import numpy as np
import struct
from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler
from qumico.handlers.frontend.tflite.tflite_decorator import tflite_op_conf
from qumico.handlers.frontend.tflitehandler import tflite_op
from qumico.handlers.frontend.tflite.padding import padding as padder
from qumico.handlers.frontend.tflite import create_property_name


#@tflite_op("DETECTION_POSTPROCESS")
#class DETECTION_POSTPROCESS(TFLiteBaseHandler):
@tflite_op("CUSTOM")
class CUSTOM(TFLiteBaseHandler):

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def custom_option_parser(*custom_options):
        options_dict = {}

        base_offset = 3
        opt_root_offset = custom_options[-base_offset]
        opt_root_type = custom_options[-2] >> 2
        opt_root_bytes = 2 ** (custom_options[-2] & 3)
        opt_root_size = custom_options[-1]

        values_offset = opt_root_offset
        types_offset = opt_root_offset // (opt_root_bytes +1)

        vector_size = 0
        key_vector_table_offset = 0
        for i in range(opt_root_bytes):
            vector_size = vector_size * 256 + custom_options[-1-opt_root_offset-base_offset-i]
            key_vector_table_offset = key_vector_table_offset * 256 + custom_options[-1-opt_root_offset-base_offset-i-8]
        key_vector_table_offset += 12 + opt_root_offset

        for i in range(vector_size):
            key_vector_offset = custom_options[-key_vector_table_offset-base_offset+i]
            key_str = ''
            for j in range(256):
                if (custom_options[-key_vector_table_offset-key_vector_offset-base_offset+i+j] == 0):
                    break
                key_str += chr(custom_options[-key_vector_table_offset-key_vector_offset-base_offset+i+j])
            values = 0
            types = custom_options[-types_offset-base_offset+i] >> 2
            bytes = 2 ** (custom_options[-types_offset-base_offset+i] & 3)
            for j in range(bytes):
                values = values * 256 + custom_options[-values_offset-base_offset+i*4+3-j]
            if (types == 3):
                values = struct.unpack('f',struct.pack('i',values))[0]
            options_dict.update({key_str: values})

        return options_dict

    @classmethod
    @tflite_op_conf([])
    def create_onnx_node(cls, operator, inputs, outputs,
                         input_buffers, output_buffers, data_format,
                         *args, **kwargs):

# Inputs
# NumInputs(node)=3
# :input_box_encodings(batch,num_boxes,4)                       float32
# :input_class_predictions(batch,num_boxes,num_classes)         float32
# :input_anchors(num_boxes,4)                                   float32
# Outputs  (batch=1)
# NumOutputs(node)=4
# :kOutputTensorDetectionBoxes(batch, num_detected_boxes, 4)    float32
# :kOutputTensorDetectionClasses(batch, num_detected_boxes)     float32
# :kOutputTensorDetectionScores(batch, num_detected_boxes)      float32
# :kOutputTensorNumDetections(1)                                float32



#        node = DETECTION_POSTPROCESS()
        node = CUSTOM()

        input_names = [i.name for i in inputs]
        output_names = [o.name for o in outputs]

        custom_option_dict = cls.custom_option_parser(*operator.custom_options)

        # inputs/outputs
        input0_name = input_names[0]    # input_box_encodings: 3D-shape(batch, ,num_boxes, 4(box_axis) )
        input1_name = input_names[1]    # input_class_predictions: 3D-shape(batch, num_boxes, num_classes_with_background)
        input2_name = input_names[2]    # input_anchors: 2D-shape(num_boxes, 4(box_axis) )
        output0_name = output_names[0]  # (boxes) 1D-shape(TopK_K[0])  selected by index
        output1_name = output_names[1]  # (classes) 1D-shape(TopK_K[0])  selected by index
        output2_name = output_names[2]  # (scores) 1D-shape(TopK_K[0])  selected by index
        output3_name = output_names[3]  # (num_detections) 1D-shape(1)

        # custom option
        NonMaxSuppression_iou_threshold = custom_option_dict['nms_iou_threshold']
        NonMaxSuppression_score_threshold = custom_option_dict['nms_score_threshold']
        NonMaxSuppression_max_detections = custom_option_dict['max_detections']
        NonMaxSuppression_num_classes = custom_option_dict['num_classes']
        NonMaxSuppression_max_classes_per_detection = custom_option_dict['max_classes_per_detection']
        NonMaxSuppression_max_categories_per_anchor = custom_option_dict['max_classes_per_detection']   # alias
        if (NonMaxSuppression_max_categories_per_anchor < NonMaxSuppression_num_classes):
            NonMaxSuppression_num_categories_per_anchor = NonMaxSuppression_max_categories_per_anchor
        else:
            NonMaxSuppression_num_categories_per_anchor = NonMaxSuppression_num_classes
        NonMaxSuppression_detections_per_class = custom_option_dict['detections_per_class']
        NonMaxSuppression_scale_values_y = custom_option_dict['y_scale']
        NonMaxSuppression_scale_values_x = custom_option_dict['x_scale']
        NonMaxSuppression_scale_values_h = custom_option_dict['h_scale']
        NonMaxSuppression_scale_values_w = custom_option_dict['w_scale']
        NonMaxSuppression_use_regular_nms = custom_option_dict['use_regular_nms']

        # todo: check quant
        # input quant
        if (inputs[0].quantization is not None):
            input0_box_scale = create_property_name(input_names[0], 'box_scale')
            input0_box_zero_point = create_property_name(input_names[0], 'box_zero_point')

            input0_scale = inputs[0].quantization.scale
            input0_zero_point = inputs[0].quantization.zero_point

            if not (inputs[0].quantization.details == 0 and
                    inputs[0].quantization.quantized_dimension == 0):
                raise ValueError('Custom Quantization not supported')

        if (inputs[1].quantization is not None):
            input1_score_scale = create_property_name(input_names[1], 'score_scale')
            input1_score_zero_point = create_property_name(input_names[1], 'score_zero_point')

            input1_scale = inputs[1].quantization.scale
            input1_zero_point = inputs[1].quantization.zero_point

            if not (inputs[1].quantization.details == 0 and
                    inputs[1].quantization.quantized_dimension == 0):
                raise ValueError('Custom Quantization not supported')

        if (inputs[2].quantization is not None):
            input2_anchor_scale = create_property_name(input_names[2], 'anchor_scale')
            input2_anchor_zero_point = create_property_name(input_names[2], 'anchor_zero_point')

            input2_scale = inputs[2].quantization.scale
            input2_zero_point = inputs[2].quantization.zero_point

            if not (inputs[2].quantization.details == 0 and
                    inputs[2].quantization.quantized_dimension == 0):
                raise ValueError('Custom Quantization not supported')

        # value_info input    input0
        node.onnx_value_infos.append(helper.make_tensor_value_info(input0_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                   inputs[0].shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info(input0_box_scale,
                                                                   TensorProto.FLOAT,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info(input0_box_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                   (1,)))

        # value_info input    input1
        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                                   inputs[1].shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_score_scale,
                                                                   TensorProto.FLOAT,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_score_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('score_dequantized',
                                                                   TensorProto.FLOAT,
                                                                   inputs[1].shape))
        score_shape = (inputs[1].shape[0], inputs[1].shape[2], inputs[1].shape[1])
        node.onnx_value_infos.append(helper.make_tensor_value_info('score',
                                                                   TensorProto.FLOAT,
                                                                   score_shape))

        # value_info input    input2
        node.onnx_value_infos.append(helper.make_tensor_value_info(input2_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                                   inputs[2].shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info(input2_anchor_scale,
                                                                   TensorProto.FLOAT,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info(input2_anchor_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                                   (1,)))

        # Box decoding
        box_centersize_shape = list(inputs[0].shape)
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_centersize',
                                                                   TensorProto.FLOAT,
                                                                   inputs[0].shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_center_indices',
                                                                   TensorProto.INT64,
                                                                   (2,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size_indices',
                                                                   TensorProto.INT64,
                                                                   (2,)))
        box_center_shape = list(box_centersize_shape)
        box_center_shape[-1] = box_center_shape[-1] //2
        box_size_shape = box_center_shape
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_center',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size',
                                                                   TensorProto.FLOAT,
                                                                   box_size_shape))


        # score decoding

        # anchor decoding
        node.onnx_value_infos.append(helper.make_tensor_value_info('anchor',
                                                                   TensorProto.FLOAT,
                                                                   inputs[2].shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_center_indices',
                                                                   TensorProto.INT64,
                                                                   (2,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_size_indices',
                                                                   TensorProto.INT64,
                                                                   (2,)))

        anchor_center_shape = list(inputs[2].shape)
        anchor_center_shape[-1] = anchor_center_shape[-1] //2
        anchor_size_shape = anchor_center_shape
        node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_center',
                                                                   TensorProto.FLOAT,
                                                                   anchor_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_size',
                                                                   TensorProto.FLOAT,
                                                                   anchor_size_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_center_mul1',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_center_mul2',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('center',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size_scale',
                                                                   TensorProto.FLOAT,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size_div',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size_exp',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size_mul',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('half',
                                                                   TensorProto.FLOAT,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_size_half',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        box_min_shape = list(box_center_shape)
        box_max_shape = list(box_center_shape)
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_min',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info('box_max',
                                                                   TensorProto.FLOAT,
                                                                   box_center_shape))
        decoded_box_shape = list(box_max_shape)
        decoded_box_shape[-1] = decoded_box_shape[-1] *2
        node.onnx_value_infos.append(helper.make_tensor_value_info('decoded_boxes',
                                                                   TensorProto.FLOAT,
                                                                   decoded_box_shape))

        # NonMaxSuppression
        node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_max_output_boxes',
                                                                   TensorProto.INT64,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('iou_threshold',
                                                                   TensorProto.FLOAT,
                                                                   (1,)))
        node.onnx_value_infos.append(helper.make_tensor_value_info('score_threshold',
                                                                   TensorProto.FLOAT,
                                                                   (1,)))

        if (NonMaxSuppression_use_regular_nms == 1):
            NonMaxSuppression_selected_indices_shape = (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes, 3)
            node.onnx_value_infos.append(helper.make_tensor_value_info('NonMaxSuppression_selected_indices',
                                                                   TensorProto.INT64,
                                                                   NonMaxSuppression_selected_indices_shape))
            score_flatten_shape = ((inputs[1].shape[0] * inputs[1].shape[2] * inputs[1].shape[1]),)
            node.onnx_value_infos.append(helper.make_tensor_value_info('score_flatten',
                                                                   TensorProto.FLOAT,
                                                                   score_flatten_shape))
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_boxes',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes, 1)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('identity_mat',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes,
                                                                    NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_classes',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes, 1)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_classes_mul',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes, 1)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('num_boxes',
                                                                   TensorProto.INT64,
                                                                   (1,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_indices',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes,1)))
            TopK_X_shape = (NonMaxSuppression_detections_per_class * NonMaxSuppression_num_classes, )
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_indices_flatten',
                                                                   TensorProto.INT64,
                                                                   TopK_X_shape))
            node.onnx_value_infos.append(helper.make_tensor_value_info('score_flatten_shape_node',
                                                                   TensorProto.INT64,
                                                                   (1,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('TopK_X',
                                                                   TensorProto.FLOAT,
                                                                   TopK_X_shape))
            node.onnx_value_infos.append(helper.make_tensor_value_info('TopK_K',
                                                                   TensorProto.INT64,
                                                                   (1,)))

            # TopK
            TopK_Values_shape = (NonMaxSuppression_max_detections,)
            node.onnx_value_infos.append(helper.make_tensor_value_info('TopK_Values',
                                                                   TensorProto.FLOAT,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('TopK_Indices',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_num_classes',
                                                                   TensorProto.INT64,
                                                                   (1,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('detected_box_indices',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,1)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('detected_box_indices_1d',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('detected_indices',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_index_org',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_index',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('anchor_index_mul',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('class_index',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('detected_classes_1d',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('detected_classes',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections,1)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('detected_scores_list',
                                                                   TensorProto.FLOAT,
                                                                   (NonMaxSuppression_max_detections,)))
            node.onnx_value_infos.append(helper.make_tensor_value_info('num_detected',
                                                                   TensorProto.INT64,
                                                                   (1,)))

        else: # (NonMaxSuppression_use_regular_nms == 0)
# tmp
#            node.onnx_value_infos.append(helper.make_tensor_value_info('sorted_score',
#                                                                   TensorProto.FLOAT,
#                                                                   score_shape))
#            node.onnx_value_infos.append(helper.make_tensor_value_info('class_indices',
#                                                                   TensorProto.INT64,
#                                                                   score_shape))
            print(score_shape)
            max_scores_shape = (inputs[1].shape[0], 1, inputs[1].shape[1])
            node.onnx_value_infos.append(helper.make_tensor_value_info('max_scores',
                                                                   TensorProto.FLOAT,
                                                                   max_scores_shape))
            NonMaxSuppression_selected_indices_shape = (NonMaxSuppression_max_detections, 3)
            node.onnx_value_infos.append(helper.make_tensor_value_info('NonMaxSuppression_selected_indices',
                                                                   TensorProto.INT64,
                                                                   NonMaxSuppression_selected_indices_shape))
            node.onnx_value_infos.append(helper.make_tensor_value_info('NMS_boxes_indices',
                                                                   TensorProto.INT64,
                                                                   (NonMaxSuppression_max_detections, )))
            selected_class_indicies_shape = (inputs[1].shape[0], inputs[1].shape[2], NonMaxSuppression_max_detections)
            node.onnx_value_infos.append(helper.make_tensor_value_info('selected_class_indices',
                                                                   TensorProto.INT64,
                                                                   selected_class_indicies_shape))

        # value_info output   output0
        node.onnx_value_infos.append(helper.make_tensor_value_info(output0_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                   (1,NonMaxSuppression_max_detections,4)))

        # value_info output   output1
        node.onnx_value_infos.append(helper.make_tensor_value_info(output1_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[1].np_tensor_type],
                                                                   (NonMaxSuppression_max_detections,1)))

        # value_info output   output2
        node.onnx_value_infos.append(helper.make_tensor_value_info(output2_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[2].np_tensor_type],
                                                                   (NonMaxSuppression_max_detections,1)))

        # value_info output   output3
        node.onnx_value_infos.append(helper.make_tensor_value_info(output3_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[3].np_tensor_type],
                                                                   (1,)))


        # input/output tensor
        node.onnx_tensors.append(helper.make_tensor(input0_box_scale,
                                                    TensorProto.FLOAT,
                                                    (1,),
                                                    input0_scale))
        node.onnx_tensors.append(helper.make_tensor(input0_box_zero_point,
                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                    (1,),
                                                    input0_zero_point))

        node.onnx_tensors.append(helper.make_tensor(input1_score_scale,
                                                    TensorProto.FLOAT,
                                                    (1,),
                                                    input1_scale))
        node.onnx_tensors.append(helper.make_tensor(input1_score_zero_point,
                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                    (1,),
                                                    input1_zero_point))

        if (len(input_buffers[2])>0):
            node.onnx_tensors.append(helper.make_tensor(input2_name,
                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                        inputs[2].shape,
                                                        input_buffers[2].tobytes()))
        node.onnx_tensors.append(helper.make_tensor(input2_anchor_scale,
                                                    TensorProto.FLOAT,
                                                    (1,),
                                                    input2_scale))
        node.onnx_tensors.append(helper.make_tensor(input2_anchor_zero_point,
                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                    (1,),
                                                    input2_zero_point))

        # node
        # operator

        # QuantizeLinear (box)
        node.onnx_nodes.append(helper.make_node('DequantizeLinear',
                                                inputs=[input0_name, input0_box_scale, input0_box_zero_point],
                                                outputs=['box_centersize'],
                                                name='box_dequantizer'))

        # QuantizeLinear (class)
        node.onnx_nodes.append(helper.make_node('DequantizeLinear',
                                                inputs=[input1_name, input1_score_scale, input1_score_zero_point],
                                                outputs=['score_dequantized'],
                                                name='score_dequantizer'))

        # Transpose (class)
        node.onnx_nodes.append(helper.make_node('Transpose',
                                                inputs=['score_dequantized'],
                                                outputs=['score'],
                                                perm=[0,2,1]))

        # QuantizeLinear (anchor)
        node.onnx_nodes.append(helper.make_node('DequantizeLinear',
                                                inputs=[input2_name, input2_anchor_scale, input2_anchor_zero_point],
                                                outputs=['anchor'],
                                                name='anchor_dequantizer'))

        # Box decoding
        # ycenter = box_centersize[0]/scale_values_y * anchor[2] + anchor[0]
        # xcenter = box_centersize[1]/scale_values_x * anchor[3] + anchor[1]
        # half_h = exp(box_centersize[2]/scale_values_h)*anchor[2]/2
        # half_w = exp(box_centersize[3]/scale_values_w)*anchor[3]/2
        # box.ymin = ycenter - half_h
        # box.xmin = xcenter - half_w
        # box.ymax = ycenter + half_h
        # box.xmax = xcenter + half_w

        node.onnx_tensors.append(helper.make_tensor('box_center_indices',
                                                    TensorProto.INT64,
                                                    (2,),
                                                    [0,1]))
        node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['box_centersize', 'box_center_indices'],
                                                outputs=['box_center'],
                                                axis = 2,
                                                name='box_center'
                                                ))
        node.onnx_tensors.append(helper.make_tensor('box_size_indices',
                                                    TensorProto.INT64,
                                                    (2,),
                                                    [2,3]))
        node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['box_centersize', 'box_size_indices'],
                                                outputs=['box_size'],
                                                axis = 2,
                                                name='box_size'
                                                ))
        node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['anchor', 'box_center_indices'],
                                                outputs=['anchor_center'],
                                                axis = 1,
                                                name='anchor_center'
                                                ))
        node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['anchor', 'box_size_indices'],
                                                outputs=['anchor_size'],
                                                axis = 1,
                                                name='anchor_size'
                                                ))

        node.onnx_nodes.append(helper.make_node('Mul',
                                                inputs=['box_center', 'anchor_size'],
                                                outputs=['box_center_mul']
                                                ))
        node.onnx_tensors.append(helper.make_tensor('box_center_scale',
                                                    TensorProto.FLOAT,
                                                    (2,),
                                                    [NonMaxSuppression_scale_values_x,NonMaxSuppression_scale_values_y]))
        node.onnx_nodes.append(helper.make_node('Div',
                                                inputs=['box_center_mul', 'box_center_scale'],
                                                outputs=['box_center_div']
                                                ))
        node.onnx_nodes.append(helper.make_node('Add',
                                                inputs=['box_center_div', 'anchor_center'],
                                                outputs=['center']
                                                ))

        node.onnx_tensors.append(helper.make_tensor('box_size_scale',
                                                    TensorProto.FLOAT,
                                                    (2,),
                                                    [NonMaxSuppression_scale_values_w,NonMaxSuppression_scale_values_h]))
        node.onnx_nodes.append(helper.make_node('Div',
                                                inputs=['box_size', 'box_size_scale'],
                                                outputs=['box_size_div']
                                                ))
        node.onnx_nodes.append(helper.make_node('Exp',
                                                inputs=['box_size_div'],
                                                outputs=['box_size_exp']
                                                ))
        node.onnx_nodes.append(helper.make_node('Mul',
                                                inputs=['box_size_exp', 'anchor_size'],
                                                outputs=['box_size_mul']
                                                ))
        node.onnx_tensors.append(helper.make_tensor('half',
                                                    TensorProto.FLOAT,
                                                    (1,),
                                                    [2,]))
        node.onnx_nodes.append(helper.make_node('Div',
                                                inputs=['box_size_mul', 'half'],
                                                outputs=['box_size_half']
                                                ))
        node.onnx_nodes.append(helper.make_node('Sub',
                                                inputs=['center', 'box_size_half'],
                                                outputs=['box_min']
                                                ))
        node.onnx_nodes.append(helper.make_node('Add',
                                                inputs=['center', 'box_size_half'],
                                                outputs=['box_max']
                                                ))
        node.onnx_nodes.append(helper.make_node('Concat',
                                                inputs=['box_min', 'box_max'],
                                                outputs=['decoded_boxes'],
                                                axis=2
                                                ))

        # NonMaxSuppression

        # input tensor
        # : regular mode
        # 1:NonMaxSuppression_boxes: 3D-shape(batch=0, spatial_dimension, 4(box_corner))
        # 2:NonMaxSuppression_scores: 3D-shape(batch=0, class, spatial_dimension)
        # 3:NonMaxSuppression_output_boxes_per_class: const 1D-shape(class)
        # 4:NonMaxSuppression_iou_threshold: const 1D-shape(1)   [node->user->intersection_over_union_threshold]
        # 5:NonMaxSuppression_score_threshold: const 1D-shape(1)   [node->user->non_max_suppression_score_threshold]
        #
        # : non-regular mode
        # 1:NonMaxSuppression_boxes: 3D-shape(batch=0, spatial_dimension, 4(box_corner))
        # 2:NonMaxSuppression_scores: 3D-shape(batch=0, class, spatial_dimension) max_scores
        # 3:NonMaxSuppression_output_boxes_per_class: const 1D-shape(class)  max_detections
        # 4:NonMaxSuppression_iou_threshold: const 1D-shape(1)   [node->user->intersection_over_union_threshold]
        # 5:NonMaxSuppression_score_threshold: const 1D-shape(1)   [node->user->non_max_suppression_score_threshold]
        #
        # attribute
        # const center_point_box=0 (corner)

        if (NonMaxSuppression_use_regular_nms == 1):
            node.onnx_tensors.append(helper.make_tensor('NMS_max_output_boxes',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [NonMaxSuppression_detections_per_class*NonMaxSuppression_num_classes]))
        else:
            node.onnx_tensors.append(helper.make_tensor('NMS_max_output_boxes',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [NonMaxSuppression_max_detections]))
        node.onnx_tensors.append(helper.make_tensor('iou_threshold',
                                                    TensorProto.FLOAT,
                                                    (1,),
                                                    [NonMaxSuppression_iou_threshold]))
        node.onnx_tensors.append(helper.make_tensor('score_threshold',
                                                    TensorProto.FLOAT,
                                                    (1,),
                                                    [NonMaxSuppression_score_threshold]))

        node.onnx_tensors.append(helper.make_tensor('NMS_num_classes',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [NonMaxSuppression_num_classes+1]))

        if (NonMaxSuppression_use_regular_nms == 1):
            node.onnx_nodes.append(helper.make_node('NonMaxSuppression',
                                                inputs=['box_centersize', 'score', 'NMS_max_output_boxes', 'iou_threshold', 'score_threshold'],
                                                outputs=['NonMaxSuppression_selected_indices'],
                                                center_point_box=0
                                                ))
        else:
            node.onnx_nodes.append(helper.make_node('TopK',
                                                inputs=['score', 'NMS_num_classes'],
                                                outputs=['sorted_score', 'class_indices'],
                                                axis=1,
                                                name='class_TopK'))
            node.onnx_nodes.append(helper.make_node('ReduceMax',
                                                inputs=['score'],
                                                outputs=['max_scores'],
                                                axes=[1,],
                                                keepdims=1
                                                ))
            node.onnx_nodes.append(helper.make_node('NonMaxSuppression',
                                                inputs=['decoded_boxes', 'max_scores', 'NMS_max_output_boxes', 'iou_threshold', 'score_threshold'],
                                                outputs=['NonMaxSuppression_selected_indices'],
                                                center_point_box=0
                                                ))

        # output tensor
        # 1:NonMaxSuppression_selected_indices: 2D-shape(num_of_detection, 3 indices(batch, class, box))


        # convert index from (batch, class, box) to linear index
        if (NonMaxSuppression_use_regular_nms == 1):
            node.onnx_tensors.append(helper.make_tensor('score_flatten_shape_node',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [(inputs[1].shape[0] * inputs[1].shape[2] * inputs[1].shape[1])]))

            node.onnx_nodes.append(helper.make_node('Reshape',
                                                inputs=['score', 'score_flatten_shape_node'],
                                                outputs=['score_flatten']
                                                ))

        # gather
            node.onnx_tensors.append(helper.make_tensor('index2',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [2]))
            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['NonMaxSuppression_selected_indices', 'index2'],
                                                outputs=['NMS_boxes'],
                                                axis=1,
                                                name='NMS_boxes'
                                                ))

            node.onnx_tensors.append(helper.make_tensor('index1',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [1]))
            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['NonMaxSuppression_selected_indices', 'index1'],
                                                outputs=['NMS_classes'],
                                                axis=1,
                                                name='NMS_classes'
                                                ))

        # new index = classes * max_num_of_boxes + boxes
            node.onnx_tensors.append(helper.make_tensor('num_boxes',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [score_shape[2]]))
        
            node.onnx_nodes.append(helper.make_node('Mul',
                                                inputs=['NMS_classes', 'num_boxes'],
                                                outputs=['NMS_classes_mul']
                                                ))
            node.onnx_nodes.append(helper.make_node('Add',
                                                inputs=['NMS_classes_mul', 'NMS_boxes'],
                                                outputs=['NMS_indices']
                                                ))

            node.onnx_nodes.append(helper.make_node('Squeeze',
                                                inputs=['NMS_indices'],
                                                outputs=['NMS_indices_flatten'],
                                                axes=[1]))

            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['score_flatten', 'NMS_indices_flatten'],
                                                outputs=['TopK_X'],
                                                axis=0,
                                                name='Gather_NMS_upper_rank_data'
                                                ))

        # output tensor
        # 1:TopK_X: 1D-shape(num_of_selection)

        # TopK

        # input tensor
        # 1:TopK_X (scores): 1D-shape(num_of_selection)
        #   select scores by index, and index should be transfered from 3axis index(batch, class, box) to linear index.   (reshape or flatten)
        # 2:TopK_K = const 1D-shape(1)  [max_detections=node_user_data->max_detections]
        #
        # attribute
        # const TopK_axis=0
        # const TopK_largest=1
        # const TopK_sorted=0 or default

            node.onnx_tensors.append(helper.make_tensor('TopK_K',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [NonMaxSuppression_max_detections]))
            node.onnx_nodes.append(helper.make_node('TopK',
                                                inputs=['TopK_X', 'TopK_K'],
                                                outputs=['TopK_Values', 'TopK_Indices'],
                                                axis=0
                                                ))

        # output tensor
        # 1:TopK_Values: 1D-shape(TopK_K[0])  scores
        # 2:TopK_Indices: 1D-shape(TopK_K[0])
            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['NMS_boxes', 'TopK_Indices'],
                                                outputs=['detected_box_indices'],
                                                axis=0,
                                                name='detected_box_indices'
                                                ))

            node.onnx_nodes.append(helper.make_node('Squeeze',
                                                inputs=['detected_box_indices'],
                                                outputs=['detected_box_indices_1d'],
                                                axes=[1]
                                                ))

            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['NMS_indices_flatten', 'TopK_Indices'],
                                                outputs=['detected_indices'],
                                                axis=0,
                                                name='detected_indices'
                                                ))

#            node.onnx_nodes.append(helper.make_node('Div',
#                                                inputs=['TopK_Indices', 'NMS_num_classes'],
#                                                outputs=['anchor_index_org']
#                                                ))
#            node.onnx_nodes.append(helper.make_node('Floor',
#                                                inputs=['anchor_index_org'],
#                                                outputs=['anchor_index']
#                                                ))
#            node.onnx_nodes.append(helper.make_node('Mul',
#                                                inputs=['anchor_index', 'NMS_num_classes'],
#                                                outputs=['anchor_index_mul']
#                                                ))
#            node.onnx_nodes.append(helper.make_node('Sub',
#                                                inputs=['TopK_Indices', 'anchor_index_mul'],
#                                                outputs=['class_index']
#                                                ))
        # input tensor
        # 1:Indices: 1D-shape(TopK_K[0])

        # gather(s)
            node.onnx_nodes.append(helper.make_node('Gather',
#                                                inputs=['decoded_boxes', 'anchor_index'],
                                                inputs=['decoded_boxes', 'detected_box_indices_1d'],
                                                outputs=[output0_name],       # boxes
                                                axis=1,
                                                name='output0:boxes_gather'
                                                ))

            node.onnx_nodes.append(helper.make_node('Div',
                                                inputs=['detected_indices', 'num_boxes'],
                                                outputs=['detected_classes_1d']
                                                ))

            node.onnx_nodes.append(helper.make_node('Unsqueeze',
                                                inputs=['detected_classes_1d'],
                                                outputs=['detected_classes'],
                                                axes=[1]
                                                ))

#        node.onnx_nodes.append(helper.make_node('Gather',
##                                                inputs=['NMS_classes', 'class_index'],
#                                                inputs=['NMS_classes', 'detected_indices'],
#                                                outputs=['detected_classes'],       # classes
#                                                axis=0,
#                                                name='output1:classes_gather'
#                                                ))

            node.onnx_nodes.append(helper.make_node('Cast',
                                                inputs=['detected_classes'],
                                                outputs=[output1_name],
                                                to=TensorProto.FLOAT,
                                                name='output1:classes_cast'
                                                ))

            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['TopK_X', 'TopK_Indices'],
                                                outputs=['detected_scores_list'],       # scores
                                                axis=0,
                                                name='output2:scores_gather'
                                                ))

            node.onnx_nodes.append(helper.make_node('Unsqueeze',
                                                inputs=['detected_scores_list'],
                                                outputs=[output2_name],       # scores
                                                axes=[1],
                                                name='output2:scores_unsqueeze'
                                                ))

            node.onnx_nodes.append(helper.make_node('Shape',
                                                inputs=['TopK_Indices'],
                                                outputs=['num_detected']
                                                ))

            node.onnx_nodes.append(helper.make_node('Cast',
                                                inputs=['num_detected'],
                                                outputs=[output3_name],
                                                to=TensorProto.FLOAT,
                                                name='output3:num_results_cast'
                                                ))

        else: # (NonMaxSuppression_use_regular_nms == 0)
        # new index = index * max_num_of_classes
            node.onnx_tensors.append(helper.make_tensor('index2',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [2]))
            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['NonMaxSuppression_selected_indices', 'index2'],
                                                outputs=['NMS_boxes_indices_tensor'],
                                                axis=1,
                                                name='NMS_boxes_indices'
                                                ))
            node.onnx_nodes.append(helper.make_node('Squeeze',
                                                inputs=['NMS_boxes_indices_tensor'],
                                                outputs=['NMS_boxes_indices'],
                                                axes=[1],
                                                name='NMS_boxes_squeeze'
                                                ))

            node.onnx_tensors.append(helper.make_tensor('num_classes',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [score_shape[1]]))
#            node.onnx_nodes.append(helper.make_node('Mul',
#                                                inputs=['NMS_boxes_indices', 'num_classes'],
#                                                outputs=['selected_class_indices']
#                                                ))

        # gather
            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['decoded_boxes', 'NMS_boxes_indices'],
                                                outputs=[output0_name],
                                                axis=1,
                                                name='output0:box'
                                                ))

            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['class_indices', 'NMS_boxes_indices'],
                                                outputs=['detected_class_list'],
                                                axis=2,
                                                name='output1:class_gather'
                                                ))
            node.onnx_tensors.append(helper.make_tensor('index0',
                                                    TensorProto.INT64,
                                                    (1,),
                                                    [0]))
            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['detected_class_list', 'index0'],
                                                outputs=['sliced_class'],
                                                axis=1,
                                                name='output1:class_sliced'
                                                ))
            node.onnx_nodes.append(helper.make_node('Squeeze',
                                                inputs=['sliced_class'],
                                                outputs=['squeezed_class'],       # class
                                                axes=[0,1],
                                                name='output1:class_squeeze'
                                                ))
            node.onnx_nodes.append(helper.make_node('Unsqueeze',
                                                inputs=['squeezed_class'],
                                                outputs=['unsqueezed_class'],
                                                axes=[1]))
            node.onnx_nodes.append(helper.make_node('Cast',
                                                inputs=['unsqueezed_class'],
                                                outputs=[output1_name],
                                                to=TensorProto.FLOAT,
                                                name='output1:classes_cast'
                                                ))

            node.onnx_nodes.append(helper.make_node('Gather',
                                                inputs=['max_scores', 'NMS_boxes_indices'],
                                                outputs=['detected_scores_list'],       # scores
                                                axis=2,
                                                name='output2:scores_gather'
                                                ))
            node.onnx_nodes.append(helper.make_node('Squeeze',
                                                inputs=['detected_scores_list'],
                                                outputs=['squeezed_scores'],       # scores
                                                axes=[0,1],
                                                name='output2:scores_squeeze'
                                                ))
            node.onnx_nodes.append(helper.make_node('Unsqueeze',
                                                inputs=['squeezed_scores'],
                                                outputs=[output2_name],
                                                axes=[1]))

            node.onnx_nodes.append(helper.make_node('Cast',
                                                inputs=['NMS_max_output_boxes'],
                                                outputs=[output3_name],
                                                to=TensorProto.FLOAT,
                                                name='output3:num_results_cast'
                                                ))
        # output tensor
        # 1: output0_name: (boxes) 1D-shape(TopK_K[0])  selected by index
        # 2: output1_name: (classes) 1D-shape(TopK_K[0])  selected by index
        # 3: output2_name: (scores) 1D-shape(TopK_K[0])  selected by index
        # 4: output3_name: (num_detections) 1D-shape(1)  = TopK_Indices.size



        return node
