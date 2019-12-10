from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import copy
import inspect
import weakref
from collections import defaultdict
from functools import wraps

from qumico.common import IS_PYTHON3
from qumico.common import get_data_format
from qumico.common import get_perm_from_formats
from qumico.common import supports_device
from .handler import Handler


class BackendHandler(Handler):
    """ This class is base backend handler class.
    All backend operator handler class MUST inherit this class.
    In backend, operator handler class's name should be pascal case of file name
    which should be snake case.
    Use ONNX operator name as class name.
    """
    OnceCalled = defaultdict(bool)
    
    _instances = set()
    NAME_GEN = defaultdict(int)

    PRAGMA_OMP = "#pragma omp parallel for"
    Quantizable =False

    TF_FUNC = None
    def __init__(self,
                 node,
                 tf_func=None,
#                  inputs=None,
                 input_tensor=None,
                 attrs=None,
                 name="",
                 c_first_cuda_only=False,
                 c_last_only=False,
#                  outputs =None ,
                 output_tensor =None ,
                 **kwargs):
        self.node = node        
        self._input_tensor = input_tensor
        self.attrs= attrs
        if not name == "":
            self._name = name
        else:
            self.NAME_GEN[self.__class__] +=1
            self._name = self.__class__.__name__ + str(self.NAME_GEN[self.__class__])

        self.c_first_cuda_only= c_first_cuda_only
        self.c_last_only = c_last_only
        self._output_tensor = output_tensor
        self._instances.add(weakref.ref(self))
        
    def reset_for_run(self):
        type(self).OnceCalled = defaultdict(bool) # reset

    @property
    def name(self):
        return self._name


    @property
    def func_name(self):
        return  "Op"+self.name


    # TO: Remove, use func_name property
    def get_func_name(self):
        return  "Op"+self.name

  
    def get_name(self):
        return  self._name    
    
    @property
    def input_tensor(self):
        return self._input_tensor

    @property
    def input_tensor_names(self):
        return self._input_tensor._fields

    @property
    def input_tensor_dict(self):
        return self._input_tensor._asdict()
    
    @property
    def input_tensor_values(self):
        return list(self._input_tensor._asdict().values())

    @property # Cacheable ?
    def input_tensor_shapes(self):
        return list(v.shape for v in self._input_tensor._asdict().values())

    @property
    def input_tensor_dtypes(self):
        return list(v.dtype for v in self._input_tensor._asdict().values())

    @property
    def input_tensor_ndims(self):
        return list(v.ndim for v in self._input_tensor._asdict().values())


    @property
    def output_tensor(self):
        return self._output_tensor

    @property
    def output_tensor_names(self):
        return self._output_tensor._fields

    @property
    def output_tensor_dict(self):
        return self._output_tensor._asdict()

    @property
    def output_tensor_values(self):
        return list(self._output_tensor._asdict().values())

    @property
    def output_tensor_shapes(self):
        return list(v.shape for v in self._output_tensor._asdict().values())

    @property
    def output_tensor_dtypes(self):
        return list(v.dtype for v in self._output_tensor._asdict().values())

    @property
    def output_tensor_ndims(self):
        return list(v.ndim for v in self._output_tensor._asdict().values())

    
    @classmethod
    def dec_generate_once(cls,resType=str):
        """Runs a function (successfully) only once.
        The running can be reset by setting the `CallOnece` attribute to False
        """
        def _dec_generate_once(f):
            @wraps(f)
            def _call_once(cls, *args, **kwargs):
                key = cls.__name__ + f.__name__
                if cls.OnceCalled[key] is False:
                    cls.OnceCalled[key] = True
                    return f(cls, *args, **kwargs)
                else:
                    return resType()
            return _call_once
        return _dec_generate_once

    
    @classmethod
    def _getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead


    @classmethod
    def searchinstance(cls, name=None):
        return list(filter(lambda s:s.name==name,cls._getinstances()))[0] 
        

    """
        Implement Methods For SubClass
    """
    @classmethod
    def instantiate(cls):
        raise NotImplementedError()


    @classmethod
    def get_param_type_name(cls):
        raise NotImplementedError()


    @classmethod
    def get_c_op_file_name(cls):
        raise NotImplementedError()

    @classmethod
    def get_c_op_include_header(cls):
        raise NotImplementedError()

    @classmethod
    def get_c_param_type(cls):
        raise NotImplementedError()

    @classmethod
    def get_op_variale_def(cls):
        pass

    def generate_c_code(self, **kwargs):
        raise NotImplementedError()


    def gen_op_variables(self, node, node_num, **kwargs):
        raise NotImplementedError()


    def gen_init_func(self, node, node_num, indent=4, **kwargs):
        raise NotImplementedError()

  
    @classmethod
    def get_attrs_processor_param(cls):
        """ Get param for attrs processor.
        :return: Dict.
        """
        return {}


    @classmethod
    def _process_attrs(cls, attrs):
        """ Private method for processing attrs.
        Param for this processor got from `get_attrs_processor_param`.
        Param is dict contains two key: `default` and `raname`.
        First add default value to attrs if key does not exist.
        Second rename key to new key.
        For example:
          attrs = {"keep_dims": True}
          param = {"default": {"axis": 1},
                   "rename": {"keep_dims": "keepdims"}}
        processed_attrs = {"axis": "1", "keepdims": True}
        :param attrs: Process target attrs.
        :return: Processed attrs.
        """
        param = {"rename": {}, "default": {}}
        param.update(cls.get_attrs_processor_param())

        for k, v in param["default"].items():
            attrs.setdefault(k, v)

        for k, new_k in param["rename"].items():
            if k in attrs:
                attrs[new_k] = attrs.pop(k)

        return attrs


    @classmethod
    def make_tensor_from_onnx_node(cls,
                                 node,
                                 tf_func=None,
                                 inputs=None,
                                 attrs=None,
                                 name="",
                                 c_first_cuda_only=False,
                                 c_last_only=False,
                                 **kwargs):
        """ Helper method to make tensor.
        :param node: OnnxNode object.
        :param tf_func: Callable Tf function. Default is cls.TF_FUNC.
        :param inputs: Inputs tensor. Default is got from node.inputs.
        :param attrs: Attributes. Default is node.attrs.
        :param name: Node name.
        :param c_first_cuda_only: If channel first is only supported by cuda.
        If true and not cuda, do pre and post transpose.
        :param c_last_only: If only channel last is support,
        do pre and post transpose.
        :param kwargs: Other args.
        :return: Tensor.
        """
        tensor_dict = kwargs.get("tensor_dict", {}) # type is dict
        tf_func = tf_func or cls.TF_FUNC
        if tf_func is None:
            raise RuntimeError("No Tensorflow function is given.")
        if inputs is None:
            inputs = [tensor_dict.get(inp, None) for inp in node.inputs]
            if attrs is None:
                attrs = copy.deepcopy(node.attrs)
        name = name or node.name
        if name != "":
            attrs["name"] = name

        if c_first_cuda_only and c_last_only:
            raise ValueError(
                "c_first_cuda_only and c_last_only can not both be True.")

        if c_first_cuda_only:
            return cls.c_first_cuda_only(tf_func, inputs, attrs)
        elif c_last_only:
            return cls.c_last_only(tf_func, inputs, attrs)

        return cls._run_tf_func(tf_func, inputs, attrs)


    @classmethod
    def c_first_cuda_only(cls, tf_func, inputs, attrs):
        """ Handle operator that channel first is only supported by CUDA.
        When using CPU, two transposes should be added.
        :param tf_func: Callable Tf function.
    :param inputs: Inputs tensor.
    :param attrs: Attributes.
    :return: Tensor.
    """
        support_cuda = supports_device("CUDA")
        if not support_cuda:
            return cls._tuck_transpose(tf_func, inputs, attrs)
        return cls._run_tf_func(tf_func, inputs, attrs)


    @classmethod
    def c_last_only(cls, tf_func, inputs, attrs):
        """ Handle operator that channel last only is supported.
        Add two transposes anyway.
        :param tf_func: Callable Tf function.
        :param inputs: Inputs tensor.
        :param attrs: Attributes.
        :return: Tensor.
        """
        storage_format, compute_format = get_data_format(len(inputs[0].get_shape()))
        compute_format = compute_format.replace("C", "") + "C"
        return cls._tuck_transpose(tf_func, inputs, attrs,
                               (storage_format, compute_format))


    @classmethod
    def _tuck_transpose(cls, tf_func, inputs, attrs, data_format=None):
        x = inputs[0]
        x_rank = len(x.get_shape())
        if not data_format:
            data_format = get_data_format(x_rank)
        pre_perm = get_perm_from_formats(data_format[0], data_format[1])
        post_perm = get_perm_from_formats(data_format[1], data_format[0])
        attrs["data_format"] = data_format[1]
        if pre_perm != list(range(x_rank)):
            x_t = tf.transpose(x, perm=pre_perm)
            y = cls._run_tf_func(tf_func, [x_t] + inputs[1:], attrs)
            y_t = tf.transpose(y, perm=post_perm)
            return y_t
        return cls._run_tf_func(tf_func, inputs, attrs)


    @classmethod
    def _run_tf_func(cls, tf_func, inputs, attrs):
        """ Run Tensorflow function.
        Use only acceptable attributes of function from attrs.
        :param tf_func: Tensorflow function.
        :param inputs: Inputs.
        :param attrs: Attributes.
        :return: Tensor.
        """
        if IS_PYTHON3:
            params = list(inspect.signature(tf_func).parameters.keys())
        else:
            # use closure to get args for function using decorator
            if tf_func.__closure__ is not None:
                params = tf_func.__closure__[1].cell_contents.args
            else:
                params = inspect.getargspec(tf_func).args

        attrs = cls._process_attrs(attrs)
        return tf_func(*inputs,
                   **dict([(p, attrs[p]) for p in params if p in attrs]))