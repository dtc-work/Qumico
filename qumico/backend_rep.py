import ctypes
import _ctypes
import pathlib
from inspect import cleandoc
from collections import defaultdict, OrderedDict
from itertools import chain
from os import path

import numpy as np

from onnx.backend.base import BackendRep

from qumico.common import c_helper, data_type
from qumico import (STD_INCLUDES, ADD_INCLUDES, QUMICO_INCLUDE, QUMICO_LIB,
                    QUMICO_TEMPLATE_PATH, QUMICO_EXPORT_ROOT_PATH, QUMICO_MAIN)
from qumico.export import Export, ExportType
from qumico.compile import node_compile


class QumicoRep(BackendRep):
    def __init__(self, out_c_path, graph=None, inputs=None, outputs=None, tensor_dict=None, initializers=None):
        super(QumicoRep, self).__init__()
        self._graph = graph
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._tensor_dict = tensor_dict or {}
        self._initializers = initializers or []
        self._compiled = False
        self._out_c_path = out_c_path
    
    @property
    def out_c_path(self):
        return self._out_c_path
    
    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
    
    @property
    def initializers(self):
        return self._initializers

    @initializers.setter
    def initializers(self, initializers):
        self._initializers = initializers 
         
    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs
  
    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs
  
    @property
    def tensor_dict(self):
        return self._tensor_dict

    @tensor_dict.setter
    def tensor_dict(self, tensor_dict):
        self._tensor_dict = tensor_dict

    @property
    def input_tensor_dict(self):
        od = OrderedDict()
        for i in self.inputs:
            od.update({i:self.tensor_dict[i]})
        return od

    @property
    def output_tensor_dict(self):
        od = OrderedDict()
        for i in self.outputs:
            od.update({i:self.tensor_dict[i]})
        return od
  
    def convert_c(self, compile=True, export_type=ExportType.C, device=None):
        export =Export(QUMICO_TEMPLATE_PATH, self._out_c_path,export_type=export_type)
        export.crete_out_path()
        export.export_ops_header(device=device)

        if self.initializers:
            if export_type==ExportType.C:
                export.export_initializers(self._generate_initializers_c())
            elif export_type==ExportType.NPY:
                export.export_initializers(self._generate_initializers_npy())

        export.export_qumico(self._generate_c_code(export_type))
        for op_name, c_code in self._generate_lib_ops(device=device).items():
            export.export_lib_ops(op_name, c_code)
        
        # compile
        if compile:
            node_compile(export.export_qumico_path, device=device)
            self._compiled =True
    
    def run(self, inputs, **kwargs):
        """ Run TensorflowRep.
        :param inputs: Given inputs.
        :param kwargs: Other args.
        :return: Outputs.
        """
        super(QumicoRep, self).run(inputs, **kwargs)

        # generate c code
        if self._compiled is False:
            self.convert_c()

        # execute
        NodeDLL = ctypes.CDLL(path.join(self._out_c_path, QUMICO_MAIN + ".so")) # convert extension
        output =np.zeros(dtype=self.graph[-1].op.output_tensor_dtypes[0], shape=self.graph[-1].op.output_tensor_shapes[0])        

        inputs_ndpointers =[]
        inputs_ndpointers.extend(self._get_ndpointers(inputs))
        inputs_ndpointers.extend(self._get_ndpointers([output]))

        NodeDLL.qumico.argtypes = inputs_ndpointers
        NodeDLL.qumico.restype = ctypes.c_int
        NodeDLL.qumico(*(inputs + [output]))

        # todo move to postrun decorator method
        _ctypes.dlclose(NodeDLL._handle) # unload dll that is cached
        for n in self.graph:
            n.op.reset_for_run()
        
        return [output]


    def _generate_initializers_c(self):
        TEMPALTE_INITIALIZER = cleandoc(
            """
            {t} {name}{dim_bracket} = {values};
            """)
        res = []
        for name in self.initializers: 
            i = self.tensor_dict[name]
            mapping = {"name": name,
                       "t":data_type.np2c(i.dtype),
                       "dim_bracket": c_helper.generate_dim_bracket(i.shape),
                       "values":c_helper.generate_c_array(i)}
            res.append(TEMPALTE_INITIALIZER.format(**mapping))
        return "\n\n".join(res)


    def _generate_initializers_npy(self):
        res = OrderedDict()
        for name in self.initializers: 
            res.update({name:self.tensor_dict[name] })
        return res


    def _generate_lib_ops(self, device=None):

        libops =defaultdict(str)

        for n in self.graph:
            libops[type(n.op).__name__] += n.op.generate_c_code(device=device) + "\n\n" # todo add property to get op type name            

        return libops


    def _get_ndpointers(self, obj):
        res = []
        if obj is None:
            pass # do nothing
        elif type(obj) == list:
            for i in obj:
                res.append(np.ctypeslib.ndpointer(dtype=i.dtype, ndim=i.ndim,
                                             shape=i.shape, flags='CONTIGUOUS'))       
        return res


    def _generate_c_code(self, export_type):
        res = []
        res.append(self._generate_include(export_type))

        if export_type == ExportType.NPY:
            res.append(self._generate_inititializers_def())

        res.append(self._generate_nodes_info_def())
        res.append(self._generate_node_params_def())
        res.append(self._generate_node_outputs_def())
        res.append(self._generate_init_func())

        if export_type == ExportType.NPY:
            res.append(self._generate_load_initializers_func())

        res.append(self._generate_run_func())
        res.append(self._generate_main_func(export_type))
        return "\n\n".join(res)


    def _generate_include(self, export_type):
        op_headers = set()
        res = []
        for n in self.graph:
            op_headers.update(list(chain(n.op.get_c_op_file_name())))
        res.extend(["#include <" + i + ">" for i in STD_INCLUDES])
        res.extend(['#include "' + QUMICO_INCLUDE + "/" + i + '"' for i in ADD_INCLUDES])
        res.extend(['#include "' + QUMICO_LIB + "/" + i + '"' for i in op_headers])
        if self.initializers:
            if export_type==ExportType.C:
                res.extend(['#include "' + QUMICO_LIB + '/initializers.c"' ])
            elif export_type==ExportType.NPY:
                res.extend(['#include "' + QUMICO_INCLUDE + '/numpy.h"' ])
                res.extend(['#include "'  + 'numpy.c"' ])

        return "\n".join(res)


    def _generate_inititializers_def(self):
        TEMPALTE_INITIALIZER = cleandoc(
            """
            {t} {name}{dim_bracket};
            NUMPY_HEADER {nph_name};
            """)
        res = ["// define initializers"]
        for name in self.initializers: 
            i = self.tensor_dict[name]
            mapping = {"name": name,
                       "nph_name": name.replace("vi_", "nph_"),
                       "t":data_type.np2c(i.dtype),
                       "dim_bracket": c_helper.generate_dim_bracket(i.shape)}
            res.append(TEMPALTE_INITIALIZER.format(**mapping))
         
        return "\n".join(res)


    def _generate_nodes_info_def(self):
        res = []
        res.append("// define Nodes")
        res.append("int NodesCnt={0};".format(str(len(self.graph)))) 
        res.append("Node Nodes[{0}];".format(str(len(self.graph)))) 
        res.append("// define Node's  Params")       
        return "\n".join(res)


    def _generate_node_params_def(self):
        res = []
        for i, n in enumerate(self.graph):
            res.append(n.op.get_param_type_name() + " " + n.node_param_name + ";")
            res.append(n.gen_node_variables(i))
        return "\n".join(res)
        

    def _generate_node_outputs_def(self):
        res = []
        res.append("// Define Outptus")

        TemplateOutputs = "{t} {name}{shape};"
        for i, n in enumerate(self.graph):
            # todo:not only one onput, multiple outputs need to be generated            
            res.append(TemplateOutputs.format(**{"t": data_type.np2c(n.op.output_tensor_dtypes[0]),
                                               "name": n.output_tensor_names[0], # output_tensor_names
                                               "shape":c_helper.generate_dim_bracket(n.op.output_tensor_shapes[0])}))
        return "\n".join(res)


    def _generate_load_initializers_func(self):

        TEMPLATE_LOAD = cleandoc(
        """
        {indent}fname_b_len = strlen("{name}.npy");
        {indent}strcpy(buf, "./{out_folder}/initializers/");
        {indent}strcat(buf, "{name}.npy");
        {indent}ret = load_from_numpy({name}, buf, {len}, &{nph_name});
        {indent}if(ret != 0){{
        {indent}    return ret;
        {indent}}}
        """)
        
        res = ["int load_initializers(const char *path) {"]
        res += ["    char buf[255];"]
        res += ["    size_t path_len;"]
        res += ["    size_t fname_w_len;"]
        res += ["    size_t fname_b_len;"]
        res += ["    int ret;"]

        for name in self.initializers:
            i = self.tensor_dict[name]
            
            mapping = {"name": name,
                       "len": int(np.prod(i.shape)),
                       "nph_name": name.replace("vi_", "nph_"),
                       "out_folder": pathlib.Path(self._out_c_path).name,
                       "indent": " " * 4}
            res.append(TEMPLATE_LOAD.format(**mapping))        
        
        res +="}"
        res +="\n"
        return "\n".join(res)


    def _generate_init_func(self):
        res = ""
        res += "void init(){"
        res += "\n"
        for i, n in enumerate(self.graph):
            res += n.generate_init_node(i)
            res += "\n"
        res +="}"
        return res
    

    def _generate_run_func(self, indent=4):
        space = " " * indent
        res = []

        signature = []
        signature.extend(c_helper.generate_param_signature(self.input_tensor_dict)) # first node
        signature.extend(c_helper.generate_param_signature(self.output_tensor_dict)) # last node
        res.append("int run({0}){{".format(",".join(signature)))

        inputs_names =None
        outputs_names = None

        for i, n in enumerate(self.graph):
            inputs_names = list(self.graph[i].op.input_tensor._asdict()) # todo: chnage self.inputs
            outputs_names =list(self.graph[i].op.output_tensor._asdict()) # todo: chnage self.outputs
            res.append(n.generate_run_func(i, inputs_names=inputs_names, outputs_names=outputs_names))

        res.append(space + "return 0;")
        res.append("}")
        return "\n".join(res)

    
    def _generate_main_func(self, export_type, indent=4):
        signature = []
        inputs_sig = c_helper.generate_param_signature(self.input_tensor_dict)

        if len(inputs_sig) != 0:
            signature.extend(inputs_sig)
        signature.extend(c_helper.generate_param_signature(self.output_tensor_dict))
 
        run_call_signature =[]
        run_call_signature.extend(list(self.input_tensor_dict)) # first node
        run_call_signature.extend(self.output_tensor_dict)# lst node
        space = " " * indent
        res = []
        res.append("int qumico({0}){{".format(",".join(signature))) # todo: fix shape of arguments
        res.append(space + "init();")

        if export_type == ExportType.NPY:
            res.append(space + "load_initializers(NULL);")

        res.append(space + "run({0});".format(",".join(run_call_signature)))
        res.append(space + "return 0;")
        res.append("}")
        return "\n".join(res)
