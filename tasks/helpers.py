from __future__ import print_function, absolute_import, division

import os
import sys
from contextlib import contextmanager
from string import Template

from invoke.exceptions import UnexpectedExit

from .config import venv, WORKSPACE


def get_env(envdict, name, default=''):
    return str(envdict.get(name)) or default


def env_is(envdict, name):
    return get_env(envdict, name) == '1'


def get_input(question):
    try:
        try:
            answer = raw_input(question)
        except NameError:
            answer = input(question)  # pylint: disable=bad-builtin
    except EOFError:
        answer = ''
    return answer


def confirm(question, enabled_by_default=True, yes=False):
    """Prompt and let the user to confirm the operation"""
    if yes:
        return True

    if enabled_by_default:
        question += ' [Y/n]: '
    else:
        question += ' [y/N]: '

    var = None
    while var is None:
        user_input_origin = get_input(question)
        user_input = user_input_origin.strip().lower()
        if user_input == 'y':
            var = True
        elif user_input == 'n':
            var = False
        elif not user_input:
            var = enabled_by_default
        else:
            print('Invalid selection: %s' % user_input_origin)
    return var


def template(src, dst, values):
    with open(src) as f:
        tpl = Template(f.read())
    with open(dst, 'w') as f:
        f.write(tpl.substitute(values))


def edit_file(ctx, filename):
    editor = get_env(os.environ, 'EDITOR', 'vim')
    ctx.run('{} {}'.format(editor, filename), pty=True)


def shell(ctx, sh=None):
    if sh is None:
        sh = get_env(os.environ, 'SHELL', 'bash')
    ctx.run('{} -i'.format(sh), pty=True)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def flatten(iterable):
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        elif isinstance(value, str):
            yield value
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator


def buildcmd(*args):
    return ' '.join(flatten(args))


@contextmanager
def wscd(ctx, relpath=''):
    class WorkSpaceWrapper(object):
        def __init__(self, ctx):
            self._ctx = ctx
            self.venv = venv(os.path.dirname(os.path.dirname(ctx.buildcfg.env.PYTHON_BIN_PATH)))

        def run(self, *args, **kwargs):
            env = kwargs.pop('env', {})
            env.update({k: str(v) for k, v in self._ctx.buildcfg.env.items()})

            return self._ctx.run(*args, env=env, **kwargs)

        def if_cuda(self, iterable):
            if env_is(self._ctx.buildcfg.env, 'TF_NEED_CUDA'):
                return iterable
            return []

    with ctx.cd(os.path.join(WORKSPACE, relpath)):
        yield WorkSpaceWrapper(ctx)


@contextmanager
def gitbr(ctx, branch):
    currentBranch = ctx.run('git rev-parse --abbrev-ref HEAD', hide=True).stdout.strip()

    if currentBranch == branch:
        yield
        return

    if currentBranch == 'HEAD':
        # detached head
        prevState = ctx.run('git rev-parse HEAD', hide=True).stdout.strip()
    else:
        # on other branch
        prevState = currentBranch

    try:
        ctx.run('git branch -D {}'.format(branch), hide=True)
    except UnexpectedExit:
        pass

    ctx.run('git checkout -b {}'.format(branch), hide=True)
    yield
    ctx.run('git checkout {}'.format(prevState), hide=True)
    ctx.run('git branch -D {}'.format(branch), hide=True)


def detect_cuda():
    cuda_path = os.environ.get('CUDA_HOME', '/usr/local/cuda')

    cuda_version = os.environ.get('TF_CUDA_VERSION', os.environ.get('CUDA_VERSION', '9.1'))
    cuda_version = '.'.join(cuda_version.split('.')[:2])

    cudnn_version = os.environ.get('TF_CUDNN_VERSION', os.environ.get('CUDNN_VERSION', '7'))
    cudnn_version = '.'.join(cudnn_version.split('.')[:1])

    return cuda_path, cuda_version, cudnn_version


def detect_executible(candidates):
    for exe in candidates:
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe
    return ''
