import string
import numpy as np
import functools
from inspect import cleandoc

from qumico.common import data_type

def dec_add_braces():
    def _dec_add_brances(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs): 
            return "{" +  func(*args, **kwargs) + "}"
        return wrapper    
    return _dec_add_brances


@dec_add_braces()
def generate_c_array_base(nparray,ends_with_linesep=False):
    join_str=",\n" if ends_with_linesep else ","
    return join_str.join(map(lambda s:str(s), nparray))


def generate_c_array(nparray):
    res = ""
    row  =[]
    if nparray.ndim == 0:
        nparray = nparray.reshape(1)

    for i in range(nparray.shape[0]):
        if nparray[i].ndim == 0:
            row =nparray
        elif nparray[i].ndim == 1:
            row.append("  " + generate_c_array_base(nparray[i]))
        else:
            row.append("\n  " + generate_c_array(nparray[i]))
    else:
        res = generate_c_array_base(row, ends_with_linesep=True)
    return res


def generate_dim_bracket(shape):
    res = ""

    if shape==tuple():
        shape = (1,)

    for s in shape:
        res += "[{0}]".format(s)
    return res


def generate_std_include(name):
    return '#include <' + name + '>'


def generate_local_include(name):
    return '#include "' + name + '"'


def generate_param_signature(param_dict, override_param_name=None):
    param_signatures = []
    param_signature = " {type} {name}{dim}" 
    
    for param_name, param_val in param_dict.items(): 
        mapping = {"type": data_type.np2c(param_val.dtype),
                   "dim": generate_dim_bracket(param_val.shape),
                   "name":param_name if override_param_name==None else override_param_name}
        param_signatures.append(param_signature.format(**mapping))
    return param_signatures


def generate_ndim_for_loop(nparray, indent=4,gen=None, res=None, pragma=False):
    res =res if res is not None else ""
    gen = gen if gen is not None else iter(string.ascii_lowercase[8:]) # start i, j, k ...

    if nparray.ndim == 1:
        val = next(gen)
        TemplateLoop = cleandoc("""
        {indent}for(int {val}=0;{val}<{limit};{val}++){{
        {indent}    {statements}
        {indent}}}
        """)
        
        mapping = {"val": val, "limit": nparray.shape[0],
                   "indent": " " * (indent + 4), "statements": "[statements]"}
        res += TemplateLoop.format(**mapping) + "\n"

    else:
        val = next(gen)
        inner = generate_ndim_for_loop(np.empty(nparray.shape[1:]), indent=indent+4, gen=gen,res=res, pragma=False)
        mapping = {"val": val,"limit": nparray.shape[0], "indent": " " * indent}

        res += "{indent}for(int {val}=0;{val}<{limit};{val}++){{\n".format(**mapping)
        if pragma:
            res += "{indent}[pragma]\n".format(**{"indent": "  " * indent})

        res += inner
        res += "{indent}}}\n".format(**{"indent": " " * indent})
    return res
