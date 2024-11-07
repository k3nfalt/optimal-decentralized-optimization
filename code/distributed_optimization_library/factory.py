class Meta(type):
    def __new__(cls, name, bases, dct):
        cls = super().__new__(cls, name, bases, dct)
        cls._registered = {}
        return cls

class Factory(object, metaclass=Meta):
    @classmethod
    def register(factory_cls, name):
        def register_(cls):
            assert name not in factory_cls._registered
            cls.FACTORY_NAME = name
            @classmethod
            def factory_name(cls):
                return cls.FACTORY_NAME
            cls.factory_name = factory_name
            factory_cls._registered[name] = cls
            return cls
        return register_
    
    @classmethod
    def get(factory_cls, name):
        return factory_cls._registered[name]
