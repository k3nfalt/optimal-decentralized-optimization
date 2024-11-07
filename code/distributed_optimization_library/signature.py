class Signature(object):
    def __init__(self, cls, *args, **kwargs):
        self._cls = cls
        self._args = args
        self._kwargs = kwargs
        
    def create_instance(self):
        args_instances = []
        for arg in self._args:
            if isinstance(arg, Signature):
                arg = arg.create_instance()
            args_instances.append(arg)
        try:
            return self._cls(*args_instances, **self._kwargs)
        except TypeError as ex:
            raise RuntimeError("Wrong arguments {} to class {}".format(
                (args_instances, self._kwargs), self._cls))
