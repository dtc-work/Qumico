import subprocess
from os import path
from qumico.device import QumicoDevice, RaspberryPi3, QumicoDeviceType
 

def node_compile(c_path, device):
    so_name = path.basename(c_path).replace(".c", ".so")

    args =[]
    # default option
    args.append("gcc")
    args.append(path.basename(c_path))
    args.append("-o" + so_name)
    args.append("-shared")
    args.append("-fPIC")
    args.append("-std=c99")
    args.append("-Ofast")
    args.append("-lm")

    # device specific
    if issubclass(device.__class__, QumicoDevice):
        args+=device.get_compile_option()
    
    p = subprocess.Popen(args=args, cwd=path.dirname(c_path))     
    p.wait()
