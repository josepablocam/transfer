from setuptools import setup

setup(
    name='wranglesearch',
    version='0.1',
    description='Mining wrangling functions from Python programs',
    url='https://github.com/josepablocam/wranglesearch',
    author='Jose Cambronero',
    author_email='jcamsan@mit.edu',
    license='MIT',
    packages=['wranglesearch'],
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
        'tqdm',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
)
