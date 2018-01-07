from distutils.core import setup
from setuptools import setup, find_packages

def check_dependencies():
    install_requires = []
    
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import tensorflow
    except ImportError:
        install_requires.append('tensorflow')
    return install_requires

install_requires = check_dependencies()

setup(
  name = 'skopos',
  # packages = ['skopos'],
  version = '0.6',
  description = 'Deep Reinforcement Learning Library',
  author = 'Skopos-team',
  packages=find_packages(),
  # packages=[package for package in find_packages() if package.startswith('skopos')],
  install_requires=install_requires,
  author_email = 'skopos.library@gmail.com',
  url = 'https://github.com/Skopos-team/Skopos', 
  license='Apache2',
  download_url = 'https://github.com/Skopos-team/Skopos/archive/0.6.tar.gz',
  keywords = ['testing', 'logging', 'example'],
  classifiers = ['Programming Language :: Python :: 2.7',
                  'Operating System :: POSIX',
                  'Operating System :: Unix',
                  'Operating System :: MacOS'
                  ],
)