from __future__ import print_function, absolute_import, division

import os
import sys
from contextlib import contextmanager
from string import Template

from .config import venv, WORKSPACE


def get_env(envdict, name):
    return envdict.get(name) or ''


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
    editor = os.getenv('EDITOR')
    if not editor:
        editor = 'vim'
    ctx.run('{} {}'.format(editor, filename), pty=True)


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
            self.venv = venv(os.path.dirname(ctx.buildcfg.env.PYTHON_BIN_PATH))

        def run(self, *args, **kwargs):
            if 'env' in kwargs:
                tmp = dict(self._ctx.buildcfg.env)
                tmp.update(kwargs['env'])
                kwargs['env'] = tmp
            else:
                kwargs['env'] = dict(self._ctx.buildcfg.env)

            return self._ctx.run(*args, **kwargs)

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
    else:
        ctx.run('git checkout -b {}'.format(branch), hide=True)
        try:
            yield
        finally:
            ctx.run('git checkout {}'.format(currentBranch), hide=True)
            ctx.run('git branch -D {}'.format(branch), hide=True)
