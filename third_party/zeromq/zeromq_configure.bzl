"""Repository rule for ZeroMQ autoconfiguration.

`zeromq_configure` depends on the following environment variables:

  * `ZEROMQ_PATH`: location of ZeroMQ binary.
"""

_ZEROMQ_PATH = "ZEROMQ_PATH"

def _fail(msg):
    """Output failure message when configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sError when configure ZeroMQ:%s %s\n" % (red, no_color, msg))

def _check_and_link_dir(repository_ctx, base_path, name):
    # Check file exists
    dir_path = base_path + "/" + name
    if not repository_ctx.path(dir_path).exists:
        _fail("Cannot find %s" % dir_path)

    # Link the lib
    repository_ctx.symlink(dir_path, name)

def _lib_name(lib, cpu_value, version="", static=False):
    """Constructs the platform-specific name of a library.

    Args:
     lib: The name of the library, such as "cudart"
     cpu_value: The name of the host operating system.
     version: The version of the library.
     static: True the library is static or False if it is a shared object.

    Returns:
     The platform-specific name of the library.
    """
    if cpu_value in ("Linux", "FreeBSD"):
        if static:
            return "lib%s.a" % lib
        else:
            if version:
                version = ".%s" % version
            return "lib%s.so%s" % (lib, version)
    elif cpu_value == "Windows":
        return "%s.lib" % lib
    elif cpu_value == "Darwin":
        if static:
            return "lib%s.a" % lib
        else:
            if version:
                version = ".%s" % version
            return "lib%s%s.dylib" % (lib, version)
    else:
        _fail("Invalid cpu_value: %s" % cpu_value)

def _find_lib(lib, repository_ctx, cpu_value, basedir, version="", static=False):
    """Finds the given CUDA or cuDNN library on the system.

    Args:            
     lib: The name of the library, such as "cudart"
     repository_ctx: The repository context.
     cpu_value: The name of the host operating system.
     basedir: The install directory of CUDA or cuDNN.
     version: The version of the library.
     static: True if static library, False if shared object.

    Returns:
     Returns a struct with the following fields:
      file_name: The basename of the library found on the system.
      path: The full path to the library.
    """
    file_name = _lib_name(lib, cpu_value, version, static)
    path = repository_ctx.path("%s/lib/%s" % (basedir, file_name))
    if path.exists:
        return struct(file_name=file_name, path=str(path.realpath))

    _fail("Cannot find library %s" % file_name)

def _cpu_value(repository_ctx):
    """Returns the name of the host operating system.

    Args:
     repository_ctx: The repository context.

    Returns:
     A string containing the name of the host operating system.
    """
    os_name = repository_ctx.os.name.lower()
    if os_name.startswith("mac os"):
        return "Darwin"
    if os_name.find("windows") != -1:
        return "Windows"
    result = repository_ctx.execute(["uname", "-s"])
    return result.stdout.strip()

def _zeromq_autoconf_impl(repository_ctx):
    path = repository_ctx.os.environ[_ZEROMQ_PATH]

    # link the include and lib
    _check_and_link_dir(repository_ctx, path, "include")
    _check_and_link_dir(repository_ctx, path, "lib")

    cpu_value = _cpu_value(repository_ctx)
    # copy over the template
    repository_ctx.template('BUILD', Label("//third_party/zeromq:BUILD.tpl"),
        {
            "${zmq_lib}": _find_lib("zmq", repository_ctx, cpu_value, path, 5).file_name,
            "${sodium_lib}": _find_lib("sodium", repository_ctx, cpu_value, path, 23).file_name
        })

zeromq_configure = repository_rule(
    implementation = _zeromq_autoconf_impl,
    environ = [
        _ZEROMQ_PATH,
    ],
)
"""Detects and configures the local ZeroMQ.

Add the following to your WORKSPACE FILE:

```python
zeromq_configure(name = "local_config_zeromq")
```

Args:
  name: A unique name for this workspace rule.
"""
