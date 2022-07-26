#!/usr/bin/env python

from distutils.core import setup

setup(name='rosetta_can',
      version='0.1.1',
      description='Python CAN Analysis to find Signals',
      author='Matthew Nice',
      author_email='matthew.nice@vanderbilt.edu',
      url='https://github.com/jmscslgroup/rosetta_can',
      classifiers=[
          "Programming Language :: Python :: 3",
          "Framework :: AsyncIO",
          "Topic :: Communications",
          "Topic :: Scientific/Engineering :: Visualization",
          "License :: OSI Approved :: MIT License",
          ],
      packages=setuptools.find_packages(),
     )
