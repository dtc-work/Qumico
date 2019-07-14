from collections import defaultdict
from onnx.backend.test import  BackendTest


class QumicoBackendTest(BackendTest):
    def __init__(self, backend, parent_module=None):
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]

        self._test_items = defaultdict(dict) # type: Dict[Text, Dict[Text, TestItem]]