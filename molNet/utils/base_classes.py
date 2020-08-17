def modification(func):
    def _wrapper(*args, **kwargs):
        args[0]._modify()
        func(*args, **kwargs)

    return _wrapper

def needs_valid(func):
    def _wrapper(*args, **kwargs):
        if not args[0].is_validated():
            args[0].validate()
        return func(*args, **kwargs)

    return _wrapper


class ValidatingObject:
    def __init__(self):
        self._validated = False

    def _modify(self):
        self.set_invalid()

    def is_validated(self):
        return self._validated

    def set_invalid(self):
        self._validated = False

    def validate(self):
        self._validated = True
        return self._validated

