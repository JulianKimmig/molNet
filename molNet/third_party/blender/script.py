import inspect

from molNet.third_party.blender.tools import BlenderFunctionCall, BlenderFunction


class BlenderScript():
    def __init__(self):
        self._blender_operations = []
        self._imports = []
        self.import_module("bpy")
        self.import_module("bmesh")
        self.import_module("numpy",as_name="np")



    def import_module(self, module_name, from_pack=None, as_name=None):
        self._imports.append(
            ("from {} ".format(from_pack) if from_pack else "") +
            ("import {} ".format(module_name)) +
            ("as {}".format(as_name) if as_name else "")
        )

    def append(self, blender_data):
        if isinstance(blender_data, BlenderFunctionCall):
            return self._blender_operations.append(blender_data)

        for obj in blender_data:
            self._blender_operations.append(obj)

    def _create_functions(self):
        functions = []

        def _add_function(bf: BlenderFunction):
            for dep in bf.dependencies:
                _add_function(dep)
            if bf not in functions:
                functions.append(bf)

        for f in self._blender_operations:
            if isinstance(f, BlenderFunctionCall):
                if f.blender_function:
                    _add_function(f.blender_function)

        return functions

    def _create_calls(self):
        calls = []
        for f in self._blender_operations:
            if isinstance(f, BlenderFunctionCall):
                calls.append(f.to_string())
            else:
                calls.append(str(f))
        return calls

    def to_script(self,with_times=False):
        s = ""
        for i in self._imports:
            s += i + "\n"
            if with_times:
                s += "from time import time\n"
                s+="_script_timer_time = time()\n"
        s += "\n"
        _script_timer_time_counter=0
        counter="_script_timer_time_new = time()\n"+\
                "print({}, _script_timer_time_new - _script_timer_time)\n"+\
                "_script_timer_time = _script_timer_time_new\n"

        for f in self._create_functions():

            s += str(f) + "\n\n"
            if with_times:
                s+=counter.format(_script_timer_time_counter)
                _script_timer_time_counter+=1

        for c in self._create_calls():
            s += str(c) + "\n"
            if with_times:
                s+=counter.format(_script_timer_time_counter)
                _script_timer_time_counter+=1
        #        for call in self._calls:
        #            s+="{} = {}({}{})\n".format(
        #                "_",
        #                call["function"].__name__,
        #                ",".join(call["args"]) + "," if len(call["args"])>0 else "",
        #                ",".join(["{}={}".format(k,v) for k,v in call["kwargs"].items()])
        #        )
        return s

    def __str__(self):
        return self.to_script(with_times=False)
