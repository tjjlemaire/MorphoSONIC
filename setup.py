#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 10:44:35

from setuptools import setup


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()


setup(name='ExSONIC',
      version='1.0',
      description='Python+NEURON implementation of **spatially extended representations** of the \
                   **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model** to \
                   compute the distributed electrical response of morphologically realistic neuron \
                   representations to acoustic stimuli, as predicted by the *intramembrane \
                   cavitation* hypothesis.',
      long_description=readme(),
      url='https://c4science.ch/diffusion/7145/',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      keywords=('SONIC NICE extended nerve axon morphology acoustic ultrasound ultrasonic \
                neuromodulation neurostimulation excitation computational model intramembrane \
                cavitation'),
      author='Théo Lemaire',
      author_email='theo.lemaire@epfl.ch',
      license='MIT',
      packages=['ExSONIC'],
      scripts=[],
      install_requires=[
          'numpy>=1.10',
          'matplotlib>=2'
          'pandas>=0.21'
      ],
      zip_safe=False)
