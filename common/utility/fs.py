import os

def mkdir(path):
    """Create dir in path and all intermediate dirs."""

    if not os.path.exists(path):
        os.makedirs(path)
    return path

def touch(path):
    """Create file in path if not already exists."""

    if not os.path.exists(path):
        open(path, 'a').close()
    return path

def redirect(path, end, delim='/'):
    """Change ending of `path` with `end`."""

    path = path.split(delim)
    end = end.split(delim)
    assert len(path) >= len(end)
    new = path[:-len(end)] + end
    return delim.join(new)

### unused

def create_dir(_dir):
    if not os.path.exists(_dir):
        mother_dir = "/".join(_dir.split("/")[0:-1])
        if not os.path.exists(mother_dir):
            create_dir(mother_dir)
        os.mkdir(_dir)
    return _dir
