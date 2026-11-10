import os
import re
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages


ROOT = Path(__file__).resolve().parent
CHARMTYLES_HOME = ROOT.parent.parent
os.chdir(ROOT)


def get_version():
    data = {}
    fname = ROOT / 'charmnumeric' / '__init__.py'
    exec(compile(fname.read_text(), str(fname), 'exec'), data)
    return data.get('__version__')


def find_charm_include():
    env_include = os.environ.get('CHARM_INCLUDE_DIR')
    if env_include:
        candidate = Path(env_include)
        if (candidate / 'charm++.h').exists():
            return str(candidate)

    for key in ('CHARM_HOME', 'CHARM_DIR', 'CHARM_ROOT'):
        value = os.environ.get(key)
        if value:
            candidate = Path(value) / 'include'
            if (candidate / 'charm++.h').exists():
                return str(candidate)

    depfiles = [
        ROOT / 'src' / 'build' / 'CMakeFiles' / 'backend_obj.dir' / 'compiler_depend.make',
        CHARMTYLES_HOME / 'src' / 'charmtyles' / 'core' / 'build' / 'CMakeFiles' / 'charmtyles_core.dir' / 'compiler_depend.make',
    ]
    pattern = re.compile(r'(/[^\s]*?/charm\+\+\.h)')
    for depfile in depfiles:
        if not depfile.exists():
            continue
        match = pattern.search(depfile.read_text())
        if match:
            return str(Path(match.group(1)).parent)

    home = Path.home()
    for candidate in (
        home / 'charm' / 'mpi-linux-x86_64' / 'include',
        home / 'charm-gpu' / 'include',
        home / 'charm' / 'include',
    ):
        if (candidate / 'charm++.h').exists():
            return str(candidate)

    raise RuntimeError(
        'Could not locate Charm++ headers. Set CHARM_HOME or CHARM_INCLUDE_DIR before building charmnumeric.'
    )


install_requires = ['numpy>=2.0', 'Cython']
tests_require = ['pytest']
docs_require = ['sphinx']

classes = '''
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development :: Libraries
Topic :: Utilities
'''
classifiers = [x.strip() for x in classes.splitlines() if x]

extensions = [
    Extension(
        'charmnumeric._native_region',
        [str(ROOT / 'charmnumeric' / '_native_region.pyx')],
        language='c++',
        include_dirs=[
            np.get_include(),
            str(ROOT / 'src'),
            str(CHARMTYLES_HOME / 'include'),
            find_charm_include(),
        ],
        extra_compile_args=['-std=c++17', '-O3', '-DNDEBUG'],
    )
]

setup(
    name='charmnumeric',
    #version=get_version(),
    author='Aditya Bhosale',
    author_email='adityapb1546@gmail.com',
    description='A framework for writing DSLs',
    long_description=(ROOT / 'README.rst').read_text(),
    license="BSD",
    #url='https://github.com/UIUC-PPL/PyProject',
    classifiers=classifiers,
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level='3', include_path=[str(CHARMTYLES_HOME)]),
    python_requires='>=3.12',
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": docs_require + tests_require,
    },
)
