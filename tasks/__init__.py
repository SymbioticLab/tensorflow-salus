from invoke import task
from contextlib import contextmanager
import os

from .config import WORKSPACE, BUILD_BRANCH, CFGENV, VENV


def get_input(question):
    try:
        try:
            answer = raw_input(question)
        except NameError:
            answer = input(question)  # pylint: disable=bad-builtin
    except EOFError:
        answer = ''
    return answer


def confirm(question, enabled_by_default=True):
    """Prompt and let the user to confirm the operation"""
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


@contextmanager
def wscd(ctx, relpath=''):
    class WorkSpaceWrapper(object):
        def __init__(self, ctx):
            self._ctx = ctx

        def run(self, *args, **kwargs):
            if 'env' in kwargs:
                tmp = {}.update(CFGENV)
                tmp.update(kwargs['env'])
                kwargs['env'] = tmp
            else:
                kwargs['env'] = CFGENV

            return self._ctx.run(*args, **kwargs)

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


@task
def repatch(ctx):
    def maybepatch(ws, patch):
        if confirm('Apply {}?'.format(patch)):
            print('Applying {}'.format(patch))
            ws.run('git apply {}'.format(patch))

    with wscd(ctx) as ws:
        maybepatch(ws, 'tools/path-zeromq.patch')
        maybepatch(ws, 'tools/debug-build.patch')
        maybepatch(ws, 'tools/path-gcc54.patch')


@task(aliases=['bb'], positional=['bazelArgs'])
def build(ctx, bazelArgs=''):
    with wscd(ctx) as ws:
        with gitbr(ctx, BUILD_BRANCH):
            cmd = ' '.join([
                'bazel', 'build', '-c opt', '--config=cuda', bazelArgs,
                '//tensorflow:libtensorflow.so',
                '//tensorflow:libtensorflow_kernels.so',
                '//tensorflow/tools/pip_package:build_pip_package'
            ])
            ws.run(cmd, pty=True)


@task(aliases=['cf'], positional=['configureArgs'])
def config(ctx, configureArgs=''):
    with wscd(ctx) as ws:
        with gitbr(ctx, BUILD_BRANCH):
            print('Running configure')
            ws.run('./configure {}'.format(configureArgs), pty=True)
            print('Done configure')


@task(pre=[build], aliases=['bbi'], default=True)
def install(ctx):
    with wscd(ctx) as ws:
        tempdir = ws.run('mktemp -d', hide=True).stdout.strip()
        try:
            ws.run('bazel-bin/tensorflow/tools/pip_package/build_pip_package {}'.format(tempdir))
            ws.run('echo $PATH')
            ws.run('{} uninstall -y tensorflow'.format(VENV.pip), warn=True)
            ws.run('{} install {}/*.whl'.format(VENV.pip, tempdir))
        finally:
            ws.run('rm -rf {}'.format(tempdir))
