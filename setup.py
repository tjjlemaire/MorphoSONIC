# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-18 14:42:52
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 10:44:35

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


setup(
    name='ExSONIC',
    version='1.0',
    description=description(),
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
    zip_safe=False
)
