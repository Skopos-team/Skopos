from distutils.core import setup, find_packages

setup(
  name = 'skopos',
  packages = ['skopos'],
  version = '0.1',
  description = 'Deep Reinforcement Learning Library',
  author = 'Skopos-team',
  packages=[package for package in find_packages() if package.startswith('skopos')],
  author_email = 'skopos.library@gmail.com',
  url = 'https://github.com/Skopos-team/Skopos', 
  download_url = 'https://github.com/Skopos-team/Skopos/archive/0.1.tar.gz',
  keywords = ['testing', 'logging', 'example'],
  classifiers = [],
)