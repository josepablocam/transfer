from setuptools import setup

setup(
    name='transfer',
    version='0.1',
    description=
    'Transfer database wrangling code by creating a queryable database of snippets',
    url='https://github.com/josepablocam/transfer-cleaning',
    author='Jose Cambronero',
    author_email='jcamsan@mit.edu',
    license='MIT',
    packages=['transfer'],
    install_requires=[
        'plpy @ https://github.com/josepablocam/python-pl/archive/v0.1.tar.gz',
        'py2neo==3.1.2',
        'tabulate==0.8.7',
        'zss==1.2.0',
        'editdistance',
        'dill==0.3.2',
        'jupyter==1.0.0',
        'beautifulsoup4==4.9.1',
        'ipython==7.18.1',
        'nbformat==5.0.7',
        'scikit-learn==0.23.2',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
)
