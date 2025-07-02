from setuptools import setup, Extension
import sys
import os
import setuptools
from setuptools.command.build_ext import build_ext

__version__ = '0.0.1'

# Force using system's libstdc++
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# List all source files
sources = [
    'qldpc_decoder.cpp',
    'pybind_wrapper.cpp'
]

# Compiler flags
extra_compile_args = [
    '-O3',
    '-std=c++14',  # Use C++14 instead of C++17 for better compatibility
    '-D_GLIBCXX_USE_CXX11_ABI=0',  # Use old ABI for better compatibility
    '-fPIC'  # Position Independent Code
]

# Linker flags
extra_link_args = [
    '-O3',
    '-L/usr/lib/x86_64-linux-gnu',  # Use system's libstdc++
    '-Wl,-rpath,/usr/lib/x86_64-linux-gnu'  # Set runtime path to system's libstdc++
]

ext_modules = [
    Extension(
        'maxlikelihooddecoder',
        sources,
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(
    name='maxlikelihooddecoder',
    version=__version__,
    author='Debin Xiang',
    author_email='db.xiang@zju.edu.cn',
    description='Max likelihood decoder for QLDPC codes',
    long_description='',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=['pybind11>=2.6.0'],
) 