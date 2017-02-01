from setuptools import setup

setup(name='FaultPrediction',
      version='0.1',
      install_requires=['numpy',
                        'sklearn',
                        'matplotlib',
                        'scipy'],
      description='Fault prediction in software engineering.',
      url='https://github.com/amritbhanu/fss16591/project/',
      author='rpotluri & amrit',
      author_email='aagrawa8@ncsu.edu',
      packages=['fault_prediction',
                'fault_prediction.learners',
                'fault_prediction.stats'])
