from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='falco-alpha',
    version=get_version('falco-alpha/__init__.py'),
    author='Author name',
    author_email='author@gmail.com',
    license='LGPLv3+',
    keywords='python project template repository package',
    url='https://github.com/nshune/falco-alpha',
    download_url='', #TODO: Add download URL
    # download_url='http://pypi.python.org/pypi/aircraft-flight
    description='An aircraft flight simulator for MDAO built in JAX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9,<3.11',
    platforms=['any'],
    install_requires=[
        'numpy',
        'vedo',
        'pint',
        'pandas',
        'scipy',
        'matplotlib',
        'sympy',
        'pytest',
        'jax',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Libraries',
    ],
)
