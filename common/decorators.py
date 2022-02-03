def SourceROOT(func):
    def _func(*args):
        import os
        from common.utility.source import source
        env = {}
        env.update(os.environ)
        env.update(source(os.environ["ROOT_SOURCE"]))        
        func(*args, env=env)
    return _func


def Subprocess(command):
    def run(r):
        def exe_cmd(*args, **kwargs):
            import sys
            from subprocess import Popen
            r(*args)
            cmd = command(*args)
            print(cmd)
            if "env" in kwargs:
                p = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True, env=kwargs["env"])
            else:
                p = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
            out, err = p.communicate()
            code = p.returncode
            if code != 0:
                raise Exception("Subprocess execution failed")
        return exe_cmd
    return run    


def Command(script_path):
    def wrapper(cmd):
        def decorated_cmd(*args):
            return script_path + " " + cmd(*args) 
        return decorated_cmd
    return wrapper

def PythonCommand(script_path):
    def wrapper(cmd):
        def decorated_cmd(*args):
            return "python3 " + script_path + " " + cmd(*args) 
        return decorated_cmd
    return wrapper


def UnpackPythonOptions(opt):
    def unpacked(*args):
        opts = opt(*args)
        opt_str = ""
        for _opt in opts:
            opt_str += "--%s '%s' " % (_opt, str(opts[_opt]))
        return opt_str
    return unpacked


def AddOutputDirectory(dir_path, ID=None):
    from common.utility.fs import create_dir
    def _output(o):
        def make_output(*args):
            _dir = dir_path
            if ID:
                import os
                _dir = os.path.join(dir_path, ID(*args))
            create_dir(_dir)
            return o(*args, directory=_dir)
        return make_output
    return _output


def DefineLocalTargets(o):
    import os
    def _targets(*args, **kwargs):
        from luigi import LocalTarget
        outputs = o(*args)
        _dir = ""
        if "directory" in kwargs:
            _dir = kwargs["directory"]
        if type(outputs)==str:
            return LocalTarget(os.path.join(_dir, outputs))
        elif type(outputs)==dict:
            tbr = {}
            for key in outputs:
                tbr[key] = LocalTarget(os.path.join(_dir, outputs[key]))
            return tbr
        else:
            raise NotImplementedError
    return _targets
