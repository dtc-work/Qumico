from enum import Enum




# define const
class BuiltinOperator(Enum): # enum originally
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    # // DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    # // NOTE(aselle): RELU_N1_TO_1 used to be called RELU1, but it was renamed
    # // since different model developers use RELU1 in different ways. Never
    # // create another op called RELU1.
    RELU_N1_TO_1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    # // TODO(aselle): Consider rename to CONCATENATE_EMBEDDINGS
    CONCAT_EMBEDDINGS = 29
    SKIP_GRAM = 30
    CALL = 31
    CUSTOM = 32
    EMBEDDING_LOOKUP_SPARSE = 33
    PAD = 34
    UNIDIRECTIONAL_SEQUENCE_RNN = 35
    GATHER = 36
    BATCH_TO_SPACE_ND = 37
    SPACE_TO_BATCH_ND = 38
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44
    STRIDED_SLICE = 45
    BIDIRECTIONAL_SEQUENCE_RNN = 46
    EXP = 47
    TOPK_V2 = 48
    SPLIT = 49
    LOG_SOFTMAX = 50
    # // DELEGATE is a special op type for the operations which are delegated to
    # // other backends.
    # // WARNING: Experimental interface, subject to change
    DELEGATE = 51
    BIDIRECTIONAL_SEQUENCE_LSTM = 52
    CAST = 53
    PRELU = 54
    MAXIMUM = 55
    ARG_MAX = 56
    MINIMUM = 57
    LESS = 58
    NEG = 59
    PADV2 = 60
    GREATER = 61
    GREATER_EQUAL = 62
    LESS_EQUAL = 63
    SELECT = 64
    SLICE = 65
    SIN = 66
    TRANSPOSE_CONV = 67
    SPARSE_TO_DENSE = 68
    TILE = 69
    EXPAND_DIMS = 70
    EQUAL = 71
    NOT_EQUAL = 72
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    SHAPE = 77
    POW = 78
    ARG_MIN = 79
    FAKE_QUANT = 80
    REDUCE_PROD = 81
    REDUCE_MAX = 82
    PACK = 83
    LOGICAL_OR = 84
    ONE_HOT = 85
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    UNPACK = 88
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    REDUCE_ANY = 91
    SQUARE = 92
    ZEROS_LIKE = 93
    FILL = 94
    FLOOR_MOD = 95
    RANGE = 96
    RESIZE_NEAREST_NEIGHBOR = 97
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    MIRROR_PAD = 100
    ABS = 101
    SPLIT_V = 102
    UNIQUE = 103
    CEIL = 104
    REVERSE_V2 = 105
    ADD_N = 106
    GATHER_ND = 107
    COS = 108
    WHERE = 109
    RANK = 110
    ELU = 111
    REVERSE_SEQUENCE = 112
    MATRIX_DIAG = 113
    QUANTIZE = 114
    MATRIX_SET_DIAG = 115


class CustomOptionsFormat(Enum): # enum
    FLEXBUFFERS = 0


class Padding(Enum): # num
    SAME = "SAME"
    VALID = "VALID"


class ActivationFunctionType(Enum):# enum
    NONE = 0
    RELU = 1
    RELU_N1_TO_1 = 2
    RELU6 = 3
    TANH = 4
    SIGN_BIT = 5    


# BuiltinOptions
class BaseBuiltinOptions: # BuiltinOptions
    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def parse(cls, builtin_options_type, json_content):
        print(cls.__subclasses__()["'builtin_options_type"])
        return 


class ReducerOptions(BaseBuiltinOptions):
    """
        keep_dims: bool;
    """
    def __init__(self, keep_dims):
        self._keep_dims = keep_dims 

    @property
    def keep_dims(self):
        return self._keep_dims


class Conv2DOptions(BaseBuiltinOptions):
    """
      padding:Padding;
      stride_w:int;
      stride_h:int;
      fused_activation_function:ActivationFunctionType;
      dilation_w_factor:int = 1;
      dilation_h_factor:int = 1;
    """
    def __init__(self, padding, stride_w, stride_h, fused_activation_function,
                 dilation_w_factor=1, dilation_h_factor=1):
        self._padding = Padding[padding]
        self._stride_w = stride_w
        self._stride_h = stride_h
        self._fused_activation_function = ActivationFunctionType[fused_activation_function]
        self._dilation_w_factor = dilation_w_factor
        self._dilation_h_factor = dilation_h_factor

    @property
    def padding(self):
        return self._padding
 
    @property
    def stride_w(self):
        return self._stride_w

    @property
    def stride_h(self):
        return self._stride_h

    @property
    def fused_activation_function(self):
        return self._fused_activation_function

    @property
    def dilation_w_factor(self):
        return self._dilation_w_factor

    @property
    def dilation_h_factor(self):
        return self._dilation_h_factor

        
class CustomOptions(BaseBuiltinOptions):
    pass

class Pool2DOptions(BaseBuiltinOptions):
    """
      padding:Padding;
      stride_w:int;
      stride_h:int;
      filter_width:int;
      filter_height:int;
      fused_activation_function:ActivationFunctionType;
    """
    def __init__(self, padding,stride_w, stride_h, 
                 filter_width, filter_height, fused_activation_function):
        self._padding = Padding[padding]
        self._stride_w = stride_w
        self._stride_h = stride_h
        self._filter_width = filter_width
        self._filter_height = filter_height
        self._fused_activation_function = ActivationFunctionType[fused_activation_function]

    @property
    def padding(self):
        return self._padding
 
    @property
    def stride_w(self):
        return self._stride_w

    @property
    def stride_h(self):
        return self._stride_h

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def fused_activation_function(self):
        return self._fused_activation_function


class DepthwiseConv2DOptions(BaseBuiltinOptions):
    """
      // Parameters for DepthwiseConv version 1 or above.
      padding:Padding;
      stride_w:int;
      stride_h:int;
      depth_multiplier:int;
      fused_activation_function:ActivationFunctionType;
      // Parameters for DepthwiseConv version 2 or above.
      dilation_w_factor:int = 1;
      dilation_h_factor:int = 1;
    """
    def __init__(self, padding, stride_w, stride_h, depth_multiplier, 
                 fused_activation_function,
                 dilation_w_factor=1, dilation_h_factor=1):
        self._padding = Padding[padding]
        self._stride_w = stride_w
        self._stride_h = stride_h
        self._depth_multiplier=depth_multiplier
        self._fused_activation_function = ActivationFunctionType[fused_activation_function]
        self._dilation_w_factor = dilation_w_factor
        self._dilation_h_factor = dilation_h_factor

    @property
    def padding(self):
        return self._padding
 
    @property
    def stride_w(self):
        return self._stride_w

    @property
    def stride_h(self):
        return self._stride_h

    @property
    def depth_multiplier(self):
        return self._depth_multiplier

    @property
    def fused_activation_function(self):
        return self._fused_activation_function

    @property
    def dilation_w_factor(self):
        return self._dilation_w_factor

    @property
    def dilation_h_factor(self):
        return self._dilation_h_factor


class SoftmaxOptions(BaseBuiltinOptions):
    """
      beta: float;
    """
    def __init__(self, beta):
        self._beta = beta

    @property
    def beta(self):
        return self._beta


class AddOptions(BaseBuiltinOptions):
    """
      fused_activation_function:ActivationFunctionType;
    """
    def __init__(self, fused_activation_function):
        self._fused_activation_function =  ActivationFunctionType[fused_activation_function]

    @property
    def fused_activation_function(self):
        return self._fused_activation_function


class MulOptions(BaseBuiltinOptions):
    """
      fused_activation_function:ActivationFunctionType;
    """
    def __init__(self, fused_activation_function):
        self._fused_activation_function =  ActivationFunctionType[fused_activation_function]

    @property
    def fused_activation_function(self):
        return self._fused_activation_function


class TransposeOptions(BaseBuiltinOptions):
    pass


class ConcatenationOptions(BaseBuiltinOptions):
    """
    fused_activation_function:  NONE|RELU|RELU6
    axis: dimension along which the concatenation is performed
    """
    def __init__(self, fused_activation_function, axis):
        self._fused_activation_function =  ActivationFunctionType[fused_activation_function]
        self._axis = axis

    @property
    def fused_activation_function(self):
        return self._fused_activation_function

    @property
    def axis(self):
        return self._axis


class ReshapeOptions(BaseBuiltinOptions):
    """
    table ReshapeOptions {
      new_shape:[int];
    }
    """

    def __init__(self, new_shape):
        self._new_shape = new_shape

    @property
    def new_shape(self):
        return self._new_shape


class ShapeOptions(BaseBuiltinOptions):
    """
      // Optional output type of the operation (int32 or int64). Defaults to int32.
      out_type : TensorType;
    """
    def __init__(self, out_type):
        self._out_type = out_type

    @property
    def out_type(self):
        return self._out_type


class FullyConnectedOptionsWeightsFormat(Enum): # enum
    DEFAULT = 0
    SHUFFLED4x16INT8 = 1


class FullyConnectedOptions(BaseBuiltinOptions):
    """
      // Parameters for FullyConnected version 1 or above.
      fused_activation_function:ActivationFunctionType;
    
      // Parameters for FullyConnected version 2 or above.
      weights_format:FullyConnectedOptionsWeightsFormat = DEFAULT;
    """
    def __init__(self, fused_activation_function, weights_format= FullyConnectedOptionsWeightsFormat.DEFAULT):
        self._fused_activation_function =  ActivationFunctionType[fused_activation_function]
        self._weights_format = FullyConnectedOptionsWeightsFormat[weights_format]

    @property
    def fused_activation_function(self):
        return self._fused_activation_function
    
    @property
    def weights_format(self):
        return self._weights_format    


class FakeQuantOptions(BaseBuiltinOptions):
    """
      // Parameters supported by version 1:
      min:float;
      max:float;
      num_bits:int;
    
      // Parameters supported by version 2:
      narrow_range:bool;
    """
    def __init__(self, min, max, num_bits, narrow_range):
        self._qmin = min
        self._qmax = max
        self._num_bits = num_bits
        self._narrow_range = narrow_range

    @property
    def qmin(self):
        return self._qmin

    @property
    def qmax(self):
        return self._qmax

    @property
    def num_bits(self):
        return self._num_bits

    @property
    def narrow_range(self):
        return self._narrow_range


class LogisticOptions(BaseBuiltinOptions):
    pass

class QuantizeOptions(BaseBuiltinOptions):
    pass


class DequantizeOptions(BaseBuiltinOptions):
    pass
