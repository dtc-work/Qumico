import inspect
import os
import pkgutil
import importlib


def create_property_name(input_name, next_node_name):
    return input_name + "/" + next_node_name


all_files = [modname for _, modname, _ in pkgutil.walk_packages(path=[os.path.split(__file__)[0]])]
clsmembers = dict()

for cls in all_files:
    mod = (importlib.import_module(f".{cls}", __name__))
    clsmembers.update(inspect.getmembers(mod, inspect.isclass))



