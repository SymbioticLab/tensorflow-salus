from __future__ import print_function, absolute_import, division

import os


class venv(object):
    def __init__(self, vdir):
        self.dir = vdir
        for name in ['pip', 'python']:
            setattr(self, name, os.path.join(self.dir, 'bin', name))


BUILD_BRANCH = 'tfbuild'

WORKSPACE = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

TASKS_DIR = os.path.realpath(os.path.dirname(__file__))
