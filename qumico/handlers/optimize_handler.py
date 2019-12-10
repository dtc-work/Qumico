from functools import wraps
from collections import defaultdict


class OptimizeHandler:
    PRAGMA_OMP = "#pragma omp parallel for"
    OnceCalled = defaultdict(bool)
    NAME_GEN = defaultdict(int)
    _instances = set()
    Quantizable =False

    def __init__(self):
        self.NAME_GEN[self.__class__] +=1
        self._name = self.__class__.__name__ + str(self.NAME_GEN[self.__class__])

    # input output
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

    # output
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
    def get_param_type_name(cls):
        raise NotImplementedError()
    
    
    @classmethod
    def validate(cls, *args, **kwargs):
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
    
    @property
    def name(self):
        return self._name

    @property
    def func_name(self):
        return  "Op"+self.name

    # TO: Remove, use func_name property
    def get_func_name(self):
        return  "Op"+self.name
    
    def generate_c_code(self, **kwargs):
        raise NotImplementedError()

    def gen_op_variables(self, node, node_num, **kwargs):
        raise NotImplementedError()

    def optimize(self, *args, **kwargs):
        raise NotImplementedError()
    
    def gen_init_func(self, node, node_num, indent=4, **kwargs):
        raise NotImplementedError()
    
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
