from typing import Optional, List, Any, Dict


class Registry(Dict):
    def __init__(self):
        super(Registry, self).__init__()

    def add_modules(self, modules: List):
        for module in modules:
            for k, v in module.__dict__.items():
                if isinstance(v, type):
                    self.__setitem__(k, v)

    def register(self, _class):
        self.__setitem__(_class.__name__, _class)

    def __setitem__(self, key: Any, item: Any):
        self.__dict__[key] = item

    def __getitem__(self, key: Any):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key: Any):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k: Any):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_: Dict):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item: Any):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)
