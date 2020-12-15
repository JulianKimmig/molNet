import inspect

import bpy

def to_arg(a):

    if isinstance(a,str):
        return "'{}'".format(a)
    if inspect.isclass(a):
        return a.__name__

    return str(a)

class BlenderFunctionCall():
    def __init__(self, blender_function, *args, **kwargs):
        self.blender_function = blender_function
        self._args = args
        self._kwargs = kwargs
        self._varname = "_"

    def set_varname(self, name):
        self._varname = name
        return self

    def __str__(self):
        return self._varname

    def to_string(self):
        return "{} = {}({}{})".format(
            self._varname,
            self.blender_function._function.__name__,
            ",".join([to_arg(a) for a in self._args]) + "," if len(self._args) > 0 else "",
            ",".join(["{}={}".format(k, to_arg(v)) for k, v in self._kwargs.items()])
        )

    def __call__(self):
        return self.blender_function._function(*self._args, **self._kwargs)


class BlenderFunction():
    def __init__(self, function, dependencies=[],flat=False):
        self._function = function
        self.dependencies = dependencies
        self._flat=flat



    def __call__(self, *args, **kwargs):
        return BlenderFunctionCall(self, *args, **kwargs)

    def __str__(self):
        lines = inspect.getsource(self._function)
        while lines.startswith("@"):
            lines = "\n".join(lines.split("\n")[1:])
        if self._flat:
            lines.split(":",maxsplit=1)[1]
        return lines


def blender_basic_script(func):
    return BlenderFunction(func)


def blender_function(dependencies=[], imports=[],flat=False):
    def to_script(func):
        return BlenderFunction(func, dependencies=dependencies,flat=flat)

    return to_script


class BlenderLambdaCall(BlenderFunctionCall):
    def __init__(self, func,set=True, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self._set = set
        self._func = func

    def to_string(self):
        if self._set:
            return "{} = {}({}{})".format(
                self._varname,
                self._func,
                ",".join([to_arg(a) for a in self._args]) + "," if len(self._args) > 0 else "",
                ",".join(["{}={}".format(k, to_arg(v)) for k, v in self._kwargs.items()])
            )
        else:
            return "{}({}{})".format(
                self._func,
                ",".join([to_arg(a) for a in self._args]) + "," if len(self._args) > 0 else "",
                ",".join(["{}={}".format(k, to_arg(v)) for k, v in self._kwargs.items()])
            )

    def __call__(self):
        return self.blender_function._function(*self._args, **self._kwargs)

def blender_lambda_var(varname,function,*args,**kwargs):
    lambadcall=BlenderLambdaCall(function,True,*args,**kwargs)
    lambadcall.set_varname(varname)
    return lambadcall

def blender_lambda_call(function,*args,**kwargs):
    lambadcall=BlenderLambdaCall(function,False,*args,**kwargs)
    return lambadcall

class LambdaString():
    def __init__(self,s):
        self._s = s

    def __str__(self):
        return self._s

def lambda_string(string):
    return LambdaString(string)