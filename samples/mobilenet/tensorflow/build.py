from os import path
import subprocess


def build(c_path, cross_compile_for_arm=False):
    so_name = path.basename(c_path).replace(".c", ".so") 
    main_cmd = "arm-linux-gnueabihf-gcc" if cross_compile_for_arm else "gcc"
#     p = subprocess.Popen([main_cmd,  path.basename(c_path),"-o" + so_name, "-shared",'-fPIC','-lm','-Ofast', "-fopenmp","-std=c99" ],
    p = subprocess.Popen([main_cmd,  path.basename(c_path),"-o" + so_name, "-shared",'-fPIC','-lm','-Ofast',"-std=c99" ],
                         cwd=path.dirname(c_path)) 
    p.wait()


if __name__ == "__main__":

    c_path = path.join(path.dirname(path.abspath("__file__")), "out_c", "qumico.c")
    build(c_path, cross_compile_for_arm=False)
    
