from invoke import task
from contextlib import contextmanager
import os

from .config import WORKSPACE, BUILD_BRANCH, CFGENV, VENV


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
        #with ctx.prefix(' '.join("{}='{}'".format(k, v) for k, v in CFGENV.items())):
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
    with wscd(ctx) as ws:
        ws.run('git apply tools/path-zeromq.patch')
        ws.run('git apply tools/debug-build.patch')
        ws.run('git apply tools/path-gcc54.patch')


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

