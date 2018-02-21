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


def _zeromq_autoconf_impl(repository_ctx):
    path = repository_ctx.os.environ[_ZEROMQ_PATH]

    # link the include and lib
    _check_and_link_dir(repository_ctx, path, "include")
    _check_and_link_dir(repository_ctx, path, "lib")

    # copy over the template
    repository_ctx.template('BUILD', Label("//third_party/zeromq:BUILD.tpl"))


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