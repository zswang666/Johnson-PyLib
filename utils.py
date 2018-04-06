import os
import importlib

def dynamic_import(module_name):
    """ Dynamically import module according to its name
            Args:
                module_name (str): the name of module to be imported, 
                    e.g. a_dir.a_subdir.a_pyfile
            Returns:
                the imported module
            Notes:
                1. Following the above example, `a_dir` and `a_subdir` should
                   both contain `__init__.py`.
            Examples:
                >> module = a_dir.a_subdir.a_pyfile
                >> an_instance = module.a_class(args) # args is the argument of the class
                >> func_out = module.a_function(inp) # inp is the input of the function
    """
    return importlib.import_module(module_name)

def validate_dir(*dir_name, **kwargs):
    """ Check and validate a directory
            Args:
                *dir_name (str / a list of str): a directory
                **kwargs:
                    auto_mkdir (bool): automatically make directories. Default: True.
            Returns:
                dir_name (str): path to the directory
            Notes:
                1. `auto_mkdir` is performed recursively, e.g. given a/b/c,
                   where a/b does not exist, it will create a/b and then a/b/c.
                2. using **kwargs is for future extension.
    """
    # parse argument
    if len(kwargs)>0:
        auto_mkdir = kwargs.pop("auto_mkdir")
        if len(kwargs):
            raise ValueError("Invalid arguments: {}".format(kwargs))
    else:
        auto_mkdir = True

    # check and validate directory
    dir_name = os.path.join(*dir_name)
    if auto_mkdir and not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

def validate_path(*path_name, **kwargs):
    """ Check and validate a path
            Args:
                *path_name (str / a list of str): a path
                **kwargs:
                    auto_mkdir (bool): automatically make directories. Default: True.
                    check_exist (bool): whether performing check on path existence. Default: False.
            Returns:
                path_name (str): the path
                 OR path_existence (bool)
            Notes:
                1. `auto_mkdir` is performed recursively, e.g. given a/b/c,
                   where a/b does not exist, it will create a/b and then a/b/c.
                2. using **kwargs is for future extension.
    """
    # parse argument
    if len(kwargs)>0:
        auto_mkdir = kwargs.pop("auto_mkdir")
        check_exist = kwargs.pop("check_exist")
        if len(kwargs):
            raise ValueError("Invalid arguments: {}".format(kwargs))
    else:
        auto_mkdir = True
        check_exist = False

    # check and validate path
    dir_name = os.path.join(*path_name[:-1])
    path_name = os.path.join(*path_name)
    if check_exist:
        return os.path.exists(path_name)
    if auto_mkdir and not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return path_name