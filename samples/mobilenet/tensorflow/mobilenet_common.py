from os import path

WIDTH = 128
HEIGHT = 128

LABEL_FILE_PATH = path.join(path.abspath(path.dirname(__file__)), "labels.txt")
LABELS = []
LABEL_CNT = None

with open(LABEL_FILE_PATH) as fin:
    LABELS = fin.read().splitlines() 
    LABEL_CNT = len(LABELS)
