from setuptools import setup

setup(name='transfer',
      version='0.1',
      description='Transfer database wrangling code',
      url='TODO',
      author='Jose Cambronero',
      author_email='jcamsan@mit.edu',
      license='MIT',
      packages=['transfer'],
      install_requires=[
          'plpy==0.1',
          'py2neo==3.1.2',
          'tabulate',
          'zss',
          'editdistance',
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False,
      )
