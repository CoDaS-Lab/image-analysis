# -*- coding: utf-8 -*-
"""
image analysis group tools
"""
from setuptools import setup, find_packages


setup(
    name='image_analysis',
    version='0.0.1',
    url='https://github.com/CoDaS-Lab/image_analysis',
    license='BSD',
    author='CoDaSLab http://shaftolab.com/',
    author_email='s@sophiaray.info',
    description='A small but fast and easy to use stand-alone template '
                'engine written in pure python.',
    long_description=__doc__,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages(exclude=['docs', 'test']),
    install_requires=['scipy >= 0.18.1', 'scikit-learn>=0.17', 'ffmpy>=0.2.0',
                      'sk-video>=1.1.7', 'scikit-image>=0.12.0', 'wget>=3.2'],
    extras_require=None,
    include_package_data=True
)
