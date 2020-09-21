# Code for Setting up the Library
from setuptools import dist, setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the required packages
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().splitlines()

# Loading version number
with open(path.join(here, 'causai', 'VERSION')) as version_file:
    version = version_file.read().strip()
    print(version)

setup(
    name='causai',
    version=version,
    description='Causality for Machine Learning',  # Required
    license='MIT',
    long_description=long_description,
    url='https://github.com/DanielhCarranza/causai',  # Optional
    download_url='https://github.com/DanielhCarranza/causai/archive/v0.4.tar.gz',
    author='DanielhCarranza, etc.',
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='causality machine-learning causal-inference graphical-model',
    packages=find_packages(exclude=['docs', 'tests']),
    python_requires='>=3.5',
    install_requires=install_requires,
    include_package_data=True,
    package_data={'causai':['VERSION']}
)