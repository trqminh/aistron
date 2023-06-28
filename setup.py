from setuptools import setup, find_packages

setup(name='aistron',
      version='0.0.1',
      author='Minh Tran',
      packages=find_packages(),
      description='An AIS library',
      license='MIT',
      install_requires=[
            'torch',
      ],
)