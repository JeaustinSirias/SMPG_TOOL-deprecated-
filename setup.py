from setuptools import setup

setup(
    name='SMPG',
    version='1.0.0',
    packages=['src'],
    url='https://github.com/JeaustinSirias/SMPG_TOOL',
    license='MIT',
    author='Jeaustin Sirias',
    author_email='jeaustin.sirias@ucr.ac.cr',
    description='A FUNCTIONAL stable python version for SMPG tool',
    install_requires=[
        'requests==3.6.9'
    ]
)
