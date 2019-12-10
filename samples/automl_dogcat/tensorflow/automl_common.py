from os import path

WIDTH = 224
HEIGHT = 224

LABEL_FILE_PATH = path.join(path.abspath(path.dirname(__file__)), "model", "dict.txt")
LABELS = []
LABEL_CNT = None

with open(LABEL_FILE_PATH) as fin:
    LABELS = fin.read().splitlines() 
    LABEL_CNT = len(LABELS)
