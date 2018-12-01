from __future__ import print_function, absolute_import, division

import os
import sys

from invoke import task
from invoke.exceptions import Exit

from .config import BUILD_BRANCH, WORKSPACE, TASKS_DIR
from .helpers import confirm, edit_file, shell, template, eprint, buildcmd, wscd, gitbr

@task
def deps(ctx):
    """Install dependencies"""
    dependencies = [
        'zeromq@4.2.2',
        'cppzmq@4.2.2'
    ]
    ctx.run(['spack', 'install'] + dependencies)
    ctx.run(['spack', 'view', '-v', '-d', 'true', 'spack-packages'] + dependencies)

    # python dependencies
    pydependencies = [
        'numpy',
    ]
    ctx.run(['pip', 'install'] + pydependencies)


@task
def init(ctx, yes=False):
    '''Initialize the project environments
        yes: Answer yes to all questions
    '''
    if 'buildcfg' in ctx:
        print('You have already initialized the build.')
        if not confirm('Do you want to redo the initialization?',
                       enabled_by_default=False, yes=yes):
            return

    target = os.path.join(WORKSPACE, 'invoke.yml')
    if os.path.exists(target):
        if not confirm('tasks.yml already exists. Do you want to overwrite?',
                       enabled_by_default=False, yes=yes):
            return

    # Collecting information
    default_values = {
        'PYTHON_BIN_PATH': '',
        'ZEROMQ_PATH': ''
    }
    # check python
    candidates = [
        os.path.expanduser('~/.local/venvs/tfbuild/bin/python'),
        ctx.run('which python', hide=True).stdout.strip(),
        sys.executable
    ]
    for pybin in candidates:
        if os.path.isfile(pybin) and os.access(pybin, os.X_OK):
            default_values['PYTHON_BIN_PATH'] = pybin
            break
    # check zeromq
    default_values['ZEROMQ_PATH'] = os.path.join(WORKSPACE, 'spack-packages')

    print('Creating tasks.yml...')
    tpl = os.path.join(TASKS_DIR, 'invoke.yml.tpl')
    template(tpl, target, default_values)

    if confirm('Do you want to edit {}?'.format(target), yes=yes):
        edit_file(ctx, target)

    print("Remember to run `inv cf' after changing environment variables.")


@task
def checkinit(ctx):
    '''Check if this project has been initialized'''
    if 'buildcfg' not in ctx.config:
        eprint("You need to run `inv init' first to initialize the project")
        raise Exit(-1)

    # make every environment variable into string
    for k, v in ctx.buildcfg.env.items():
        ctx.buildcfg.env[k] = str(v) if v is not None else ''


@task(pre=[checkinit])
def env(ctx):
    '''Print out currently initizliaed environment variables'''
    for k, v in ctx.buildcfg.env.items():
        print('{}: {}'.format(k, repr(v)))


@task(pre=[checkinit])
def patch(ctx):
    '''Apply patch to the build system'''
    def maybepatch(ws, patch):
        if confirm('Apply {}?'.format(patch)):
            print('Applying {}'.format(patch))
            ws.run('git apply {}'.format(patch))

    with wscd(ctx) as ws:
        maybepatch(ws, 'tools/debug-build.patch')


@task(pre=[checkinit], aliases=['bb'], positional=['bazelArgs'])
def build(ctx, bazelArgs=''):
    ba = bazelArgs
    if not bazelArgs:
        if 'bazelArgs' in ctx.buildcfg:
            ba = ctx.buildcfg.bazelArgs
            if ba is None:
                ba = ''
    with wscd(ctx) as ws:
        with gitbr(ctx, BUILD_BRANCH):
            cmd = buildcmd(
                'bazel', 'build', '-c opt',
                ws.if_cuda('--config=cuda'),
                ba,
                '//tensorflow:libtensorflow_framework.so',
                '//tensorflow:libtensorflow_kernels.so',
                '//tensorflow/tools/pip_package:build_pip_package'
            )
            print(cmd)
            ws.run(cmd, pty=True)


@task(pre=[checkinit], aliases=['cf'], positional=['configureArgs'])
def config(ctx, configureArgs=''):
    with wscd(ctx) as ws:
        with gitbr(ctx, BUILD_BRANCH):
            print('Running configure')
            ws.run('./configure {}'.format(configureArgs), pty=True)
            print('Done configure')


@task(pre=[checkinit, build], aliases=['bbi'], default=True)
def install(ctx, save=False):
    with wscd(ctx) as ws:
        try:
            tempdir = ws.run('mktemp -d', hide=True).stdout.strip()
            ws.run('bazel-bin/tensorflow/tools/pip_package/build_pip_package {}'.format(tempdir))
            ws.run('{} uninstall -y tensorflow'.format(ws.venv.pip), warn=True)
            ws.run('{} install {}/*.whl'.format(ws.venv.pip, tempdir))
            if save:
                ws.run('mkdir -p {}'.format(os.path.expanduser('~/downloads')))
                ws.run('cp -a {}/* {}'.format(tempdir, os.path.expanduser('~/downloads/')))
        finally:
            if tempdir:
                ws.run('rm -rf {}'.format(tempdir))


@task(pre=[checkinit], aliases=['shell'])
def interactive(ctx, sh=None):
    with wscd(ctx) as ws:
        with gitbr(ws, BUILD_BRANCH):
            print("Entering interactive shell...")
            shell(ws, sh)


@task(pre=[checkinit, build])
def docker(ctx):
    """Populate the docker context directory by copying files over,
       preserving symlinks internal to the context directory
    """
    docker_ctx_dir = 'docker'
    with wscd(ctx, docker_ctx_dir) as ws:
        # generate wheel package
        ws.run('bazel-bin/tensorflow/tools/pip_package/build_pip_package .')

        # cp all files from bazel-output to docker context, preserving symlink
        ws.run(['mkdir', 'tensorflow-src'])
        ws.run(['cp', '-rL', 'bazel-out', 'tensorflow-src'])
        ws.run(['cp', '-rL', 'bazel-bin', 'tensorflow-src'])

