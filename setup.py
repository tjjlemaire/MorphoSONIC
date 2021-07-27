# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:53:23
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 10:44:35

import os
from setuptools import setup

readme_file = 'README.md'


def readme():
    with open(readme_file, encoding="utf8") as f:
        return f.read()


def description():
    with open(readme_file, encoding="utf8") as f:
        started = False
        lines = []
        for line in f:
            if not started:
                if line.startswith('# Description'):
                    started = True
            else:
                if line.startswith('#'):
                    break
                else:
                    lines.append(line)
    return ''.join(lines).strip('\n')


def getFiles(path):
    return [f'{path}/{x}' for x in os.listdir(path)]


setup(
    name='MorphoSONIC',
    version='1.0',
    description=description(),
    long_description=readme(),
    url='https://github.com/tjjlemaire/MorphoSONIC',
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
    packages=['MorphoSONIC'],
    scripts=getFiles('scripts') + getFiles('tests') + getFiles('examples'),
    install_requires=[
        'numpy>=1.10',
        'matplotlib>=2'
        'pandas>=0.21',
        'boltons>=20.1.0'
    ],
    zip_safe=False
)
