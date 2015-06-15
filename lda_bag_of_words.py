# this is basically a mutable named tuple
# see http://stackoverflow.com/questions/5227839/why-python-does-not-support-record-type-i-e-mutable-namedtuple
from collections import OrderedDict


class BagOfWord(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(BagOfWord, self).__init__(*args, **kwargs)
        self._initialized = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            super(BagOfWord, self).__setitem__(name, value)
        else:
            super(BagOfWord, self).__setattr__(name, value)