from inspect import cleandoc
import logging
import math

import numpy as np

from onnx.backend.base import namedtupledict
from qumico.common import c_helper
from qumico.common import data_type
from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op


@onnx_op('NonMaxSuppression')
class NonMaxSuppression(BackendHandler):

    OpenMP=False


    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]
        attrs = node.attrs

        if (attrs.get('center_point_box') == None):       # define center_point_box. default is 0.
            attrs['center_point_box'] = 0

        if (len(node.input_tensor)<3):
            max_output_boxes_per_class = 0
        else:
            max_output_boxes_per_class = node.input_tensor[2][0]

        outputs_dtype = np.int64
        outputs_shape = tuple((max_output_boxes_per_class,3))

        try:
            outputs_shape_tmp = np.ones(shape=tuple((max_output_boxes_per_class,3)), dtype=outputs_dtype)
        except Exception as e:
            logging.warn('use model output shape in NonMaxSuppression op because of shape error:{0}'.format(e))
            outputs_shape = node.outputs_info[0][1]

        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and 
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True
        
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=attrs)
    

    @classmethod
    def get_param_type_name(cls):
        return 'NonMaxSuppression'


    @classmethod
    def get_c_op_file_name(cls):
        return ['nonmaxsuppression.c']


    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ["stdlib.h"]
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            '''
            typedef struct {{
                char* name;
                int   center_point_box;
            }} NonMaxSuppression;

            float nonmaxsuppression_compute_iou( const float *decoded_boxes, const int i, const int j) {{
                const float box_iy1 = *(decoded_boxes + i*4 +0);
                const float box_ix1 = *(decoded_boxes + i*4 +1);
                const float box_iy2 = *(decoded_boxes + i*4 +2);
                const float box_ix2 = *(decoded_boxes + i*4 +3);
                const float box_jy1 = *(decoded_boxes + j*4 +0);
                const float box_jx1 = *(decoded_boxes + j*4 +1);
                const float box_jy2 = *(decoded_boxes + j*4 +2);
                const float box_jx2 = *(decoded_boxes + j*4 +3);
//                printf("(%d)%f:%f:%f:%f  (%d)%f:%f:%f:%f\\n", i, box_iy1,box_ix1,box_iy2,box_ix2, j, box_jy1,box_jx1,box_jy2,box_jx2);

                const float area_i = (box_iy2 - box_iy1) * (box_ix2 - box_ix1);
                const float area_j = (box_jy2 - box_jy1) * (box_jx2 - box_jx1);
                if (area_i <= 0 || area_j <= 0) {{ return 0.0; }}
                const float intersection_ymin = (box_iy1 > box_jy1) ? box_iy1 : box_jy1;
                const float intersection_xmin = (box_ix1 > box_jx1) ? box_ix1 : box_jx1;
                const float intersection_ymax = (box_iy2 < box_jy2) ? box_iy2 : box_jy2;
                const float intersection_xmax = (box_ix2 < box_jx2) ? box_ix2 : box_jx2;
                const float intersection_area =
                    (((intersection_ymax - intersection_ymin) >0) ? (intersection_ymax - intersection_ymin) : 0.0)
                    * (((intersection_xmax - intersection_xmin) >0) ? (intersection_xmax - intersection_xmin) : 0.0);
//                printf("intersection_area: %f\\n", intersection_area);
//                printf("iou: %f\\n", intersection_area/ (area_i + area_j - intersection_area));
                return (intersection_area / (area_i + area_j - intersection_area));
            }}

            int nonmaxsuppression_num_cmp(const void *a, const void *b) {{
                return *(int *)b - *(int *)a;
            }}

            int nonmaxsuppression_idx_sort(float score[], int indices[], int num_scores_kept) {{
                for (int i=0; i<num_scores_kept; i++) {{
                    for (int j=i+1; j<num_scores_kept; j++) {{
                        if (score[i] < score[j]) {{
                            int tmp_idx = indices[i];
                            indices[i] = indices[j];
                            indices[j] = tmp_idx;
                            float tmp_score = score[i];
                            score[i] = score[j];
                            score[j] = tmp_score;
                        }}
//                        if (score[indices[i]] < score[indices[j]]) {{
//                            int tmp_idx = indices[i];
//                            indices[i] = indices[j];
//                            indices[j] = tmp_idx;
//                        }}
                    }}
                }}
            }}
            '''
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_op_variale_def(cls):
        res = '// get_op_variale_def'
        return res

    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        TemplateStatements = '''
            const int  center_point_box = {center_point_box};
            const int  boxes_batch = {boxes_batch};
            const int  boxes_spatial = {boxes_spatial};
            const int  boxes_params = {boxes_params};
            const int  scores_batch = {scores_batch};
            const int  scores_class = {scores_class};
            const int  scores_spatial = {scores_spatial};
            const long long int  max_output_boxes_p_class = max_output_boxes_per_class[0];
            const float  iou_th = iou_threshold[0];
            const float  score_th = score_threshold[0];
            const int  selected_indices_num = {selected_indices_num};
            const int  selected_indices_index = {selected_indices_index};
//            printf("max_output: %ld\\n", max_output_boxes_per_class[0]);
//            printf("iou_th: %f\\n", iou_threshold[0]);
//            printf("score_th: %f\\n", score_threshold[0]);
//            printf("max_output: %ld\\n", max_output_boxes_p_class);
//            printf("iou_th: %f\\n", iou_th);
//            printf("score_th: %f\\n", score_th);

            memset( (void *)selected_indices, 0, sizeof({t}) * selected_indices_num * selected_indices_index );
            if (max_output_boxes_p_class == 0) {{ return; }}

            for (int n=0; n<scores_batch; n++) {{
                for (int c=0; c<scores_class; c++) {{

                    float   decoded_boxes[boxes_spatial][4];

//  decode boxes
                    if (center_point_box == 0) {{
                        for (int i=0; i<boxes_spatial; i++) {{
                            if (boxes[n][i][0] < boxes[n][i][2]) {{
                                decoded_boxes[i][0] = boxes[n][i][0];  // ymin
                                decoded_boxes[i][2] = boxes[n][i][2];  // ymax
                            }} else {{
                                decoded_boxes[i][0] = boxes[n][i][2];  // ymin
                                decoded_boxes[i][2] = boxes[n][i][0];  // ymax
                            }}
                            if (boxes[n][i][1] < boxes[n][i][3]) {{
                                decoded_boxes[i][1] = boxes[n][i][1];  // xmin
                                decoded_boxes[i][3] = boxes[n][i][3];  // xmax
                            }} else {{
                                decoded_boxes[i][1] = boxes[n][i][3];  // xmin
                                decoded_boxes[i][3] = boxes[n][i][1];  // xmax
                            }}
                        }}
                    }} else {{
                        for (int i=0; i<boxes_spatial; i++) {{
                            decoded_boxes[i][0] = boxes[n][i][1] - boxes[n][i][3]/2;  // ymin
                            decoded_boxes[i][1] = boxes[n][i][0] - boxes[n][i][2]/2;  // xmin
                            decoded_boxes[i][2] = boxes[n][i][1] + boxes[n][i][3]/2;  // ymax
                            decoded_boxes[i][3] = boxes[n][i][0] + boxes[n][i][2]/2;  // xmax
                        }}
                    }}

//
//      SelectDetectionsAboveScoreThreshold(scores, non_max_suppression_score_threshold, &keep_scores, &keep_indices);
//
                    float   keep_scores[scores_spatial];
                    int     keep_indices[scores_spatial];

                    int  num_scores_kept = 0;
                    for (int i=0; i<scores_spatial; i++) {{
                        keep_scores[i] = 0.0;
                        keep_indices[i] = 0;
                    }}
                    for (int i=0; i<scores_spatial; i++) {{
//                        printf("scores(%f)[%d:%d:%d]:th(%f) -> %d\\n", scores[n][c][i], n, c, i, score_th, num_scores_kept);
                        if (scores[n][c][i] >= score_th) {{
                            keep_scores[num_scores_kept] = scores[n][c][i];
                            keep_indices[num_scores_kept] = i;
                            num_scores_kept++;
                        }}
                    }}
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("keep_indices[%d] = %d\\n", i, keep_indices[i]);
//                    }}

//
//      DecreasingPartialArgSort(keep_scores.data(), num_scores_kept, num_scores_kept, sorted_indices.data());
//
                    int   sorted_indices[num_scores_kept];

                    for (int i=0; i<num_scores_kept; i++) {{
                        sorted_indices[i] = keep_indices[i];
//                        printf("keep_score[%d] = %f\\n", sorted_indices[i], keep_scores[i]);
                    }}
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("pre_keep_score[%d] = %f\\n", i, keep_scores[i]);
//                    }}
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("pre_keep_indices[%d] = %d\\n", i, keep_indices[i]);
//                    }}
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("pre_sorted_indices[%d] = %d\\n", i, sorted_indices[i]);
//                    }}
//                    qsort( keep_scores, num_scores_kept, sizeof(float), nonmaxsuppression_num_cmp );
                    nonmaxsuppression_idx_sort( keep_scores, sorted_indices, num_scores_kept);
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("keep_score[%d] = %f\\n", i, keep_scores[i]);
//                    }}
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("keep_indices[%d] = %d\\n", i, keep_indices[i]);
//                    }}
//                    for (int i=0; i<num_scores_kept; i++) {{
//                        printf("sorted_indices[%d] = %d\\n", i, sorted_indices[i]);
//                    }}
                    const int num_boxes_kept = num_scores_kept;
                    const int output_size = (num_scores_kept < max_output_boxes_p_class) ? num_scores_kept: max_output_boxes_p_class;
                    int num_active_candidate = num_boxes_kept;
//                    printf("output_size: %d\\n", output_size);
//                    printf("selected_indices_num: %d\\n", selected_indices_num);
//                    printf("num_active_candidate: %d\\n", num_active_candidate);

                    int     active_box_candidate[num_scores_kept];

                    for (int row=0; row<num_boxes_kept; row++) {{
                        active_box_candidate[row] = 1;
                    }}

                    int     selected[scores_spatial];
                    int     selected_box_cnt = 0;

                    for (int i=0; i<num_boxes_kept; i++) {{
                        selected[i] = -1;
                    }}
                    for (int i=0; i<num_boxes_kept; i++) {{
                        if (num_active_candidate == 0 || selected_box_cnt >= output_size) break;
                        if (active_box_candidate[i] == 1) {{
//                            selected[selected_box_cnt] = keep_indices[sorted_indices[i]];
                            selected[selected_box_cnt] = sorted_indices[i];
//                            printf("%d: selected[%d] = %d\\n", i, selected_box_cnt, selected[selected_box_cnt]);
                            selected_box_cnt++;
                            active_box_candidate[i] = 0;
                            num_active_candidate--;
                        }} else {{
                            continue;
                        }}
                        for (int j = i + 1; j < num_boxes_kept; ++j) {{
                            if (active_box_candidate[j] == 1) {{
//                                float iou = nonmaxsuppression_compute_iou((float *)decoded_boxes, keep_indices[sorted_indices[i]], keep_indices[sorted_indices[j]]);
                                float iou = nonmaxsuppression_compute_iou((float *)decoded_boxes, sorted_indices[i], sorted_indices[j]);
//                                printf("%d:%d: iou = [%f:%f:%f:%f] [%f:%f:%f:%f] %f(th:%f)\\n", i, j,
//                                    decoded_boxes[i][0], decoded_boxes[i][1], decoded_boxes[i][2], decoded_boxes[i][3],
//                                    decoded_boxes[j][0], decoded_boxes[j][1], decoded_boxes[j][2], decoded_boxes[j][3],
//                                    iou, iou_th);
                                if (iou > iou_th) {{
                                    active_box_candidate[j] = 0;
                                    num_active_candidate--;
                                }}
                            }}
                        }}
                    }}
//                    for (int i=0; i<num_boxes_kept; i++) {{
//                        printf("sorted[%d] = %d : keep_indices = %d\\n", i, sorted_indices[i], keep_indices[sorted_indices[i]]);
//                    }}
//                    for (int i=0; i<num_boxes_kept; i++) {{
//                        printf("selected[%d] = %d\\n", i, selected[i]);
//                    }}
                    int num_batch_elements = selected_indices_num/scores_batch;
                    for (int i=0; i<num_batch_elements/scores_class; i++) {{
                        if (selected[i] == -1) {{
                            selected_indices[n*num_batch_elements+c*num_batch_elements/scores_class+i][0] = -1;
                            selected_indices[n*num_batch_elements+c*num_batch_elements/scores_class+i][1] = -1;
                            selected_indices[n*num_batch_elements+c*num_batch_elements/scores_class+i][2] = -1;
                        }} else {{
                            selected_indices[n*num_batch_elements+c*num_batch_elements/scores_class+i][0] = n;
                            selected_indices[n*num_batch_elements+c*num_batch_elements/scores_class+i][1] = c;
                            selected_indices[n*num_batch_elements+c*num_batch_elements/scores_class+i][2] = selected[i];
                        }}
//                        printf("NMS_result: %d:%d:%d\\n", n, c, selected[i]);
                    }}
                }}
            }}
        '''
        mapping = {}
        mapping.update({'op_func_name': self.get_func_name()})
        mapping.update({'center_point_box': self.attrs['center_point_box']})
        mapping.update({'boxes_batch': self.input_tensor_shapes[0][0]})
        mapping.update({'boxes_spatial': self.input_tensor_shapes[0][1]})
        mapping.update({'boxes_params': self.input_tensor_shapes[0][2]})
        mapping.update({'scores_batch': self.input_tensor_shapes[1][0]})
        mapping.update({'scores_class': self.input_tensor_shapes[1][1]})
        mapping.update({'scores_spatial': self.input_tensor_shapes[1][2]})
        mapping.update({'max_output_boxes_per_class': self.input_tensor_shapes[2][0]})
        mapping.update({'iou_threshold': self.input_tensor_shapes[3][0]})
        mapping.update({'score_threshold': self.input_tensor_shapes[4][0]})
        mapping.update({'selected_indices_num': self.output_tensor_shapes[0][0]})
        mapping.update({'selected_indices_index': self.output_tensor_shapes[0][1]})
        mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})


        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, float boxes{dims_boxes}, float scores{dims_scores}, long long int max_output_boxes_per_class[], float iou_threshold[], float score_threshold[], {t} selected_indices{dims_selected_indices}, void *inputs_params, void* outputs_params) {{
            {statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'boxes': self.input_tensor_names[0]})
        mappingf.update({'dims_boxes': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({'scores': self.input_tensor_names[1]})
        mappingf.update({'dims_scores': c_helper.generate_dim_bracket(self.input_tensor_shapes[1])}) 
        mappingf.update({'max_output_boxes_per_class': self.input_tensor_names[2]})
        mappingf.update({'iou_threshold': self.input_tensor_names[3]})
        mappingf.update({'score_threshold': self.input_tensor_names[4]})
        mappingf.update({'selected_indices': self.output_tensor_names[0]})
        mappingf.update({'dims_selected_indices': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'statements': TemplateStatements.format(**mapping)})
        res += '\n\n'
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_op_variables(self, node, node_num, **kwargs):
        TemplateVariavbles = cleandoc('''
            int OpShapeNode{node_num}[] = {{{shape}}};
            int OutputShapeNode{node_num}[] = {{{shape}}};
            ''')
        ndim = self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]
        mapping = {}
        mapping.update({'shape': ','.join(map(str,shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc('''
        {indent}// define input & output
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
        {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        ''')
        
        mapping = {}
        mapping.update({'node_param_name': node.node_param_name})
        mapping.update({'node_num': str(node_num)})
        mapping.update({'add_name': self.get_name()})
        mapping.update({'ndim':str(self.output_tensor_ndims[0])})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent':' ' * indent})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def version_10(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
