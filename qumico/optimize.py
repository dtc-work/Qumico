import qumico.common.handler_helper as helper

class Optimize:
    def __init__(self, options=[]):
        self._options = []
        for o in options:
            if o in helper.get_all_optimize_handers():
                self._options.append(o)
            else:
                raise ValueError('Optimze Not Supported ')

    @property
    def options(self):
        return self._options
    