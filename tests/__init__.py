import io
import os
import shutil

from contextlib import redirect_stdout

IS_CIRCLE_CI = "CI" in os.environ and os.environ["CI"] == "true"


def is_dir_contains(dirs, file_list=None, folder_list=None):

    if file_list is None:
        file_list = []

    if folder_list is None:
        folder_list = []

    for cur_dir, sub_dirs, files in os.walk(dirs):
        file_list = [x for x in file_list if x not in files]
        folder_list = [x for x in folder_list if x not in sub_dirs]

    return not file_list and not folder_list


def remove_folder(dirs):
    try:
        shutil.rmtree(dirs)
    except FileNotFoundError:
        pass


def execute_file(filepath):
    global_namespace = {"__file__": filepath, "__name__": "__main__"}
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), global_namespace)


def read_from_output(function):
    with io.StringIO() as buf, redirect_stdout(buf):
        function()
        return buf.getvalue()

