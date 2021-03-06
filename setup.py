"""
Modified from https://github.com/openai/baselines/blob/tf2/main.py
which is licensed under the MIT License and copyright (c) 2017 OpenAI
"""
import pkg_resources
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This package is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


extras = {
    'test': [
        'pytest',
    ],
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras['all'] = all_deps

with open('CONTRIBUTORS', 'r') as f:
    authors = ', '.join(map(lambda s: s.strip(), f.read().split('\n')))

with open('requirements.txt', 'r') as f:
    pkgs = [
        s.strip().split('#', 1)[0]
        for s in f.read().split('\n') if s.strip() and s.strip()[0] != '#'
    ]

setup(name='muzero',
      packages=[package for package in find_packages()
                if package.startswith('muzero')],
      install_requires=pkgs,
      extras_require=extras,
      description='A commented and documented implementation of MuZero based on the Google DeepMind paper (Nov 2019)',
      author=authors,
      url='https://github.com/expz/muzero-general',
      author_email='jskowera@gmail.com',
      version='1.0.0')

# ensure there is some tensorflow build with version above 2.0
tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version above 2.0'
