from setuptools import setup, find_packages

setup(name='aistron',
      version='0.1.0',
      author='Minh Tran',
      packages=find_packages(),
      description='An AIS library',
      license='Apache License 2.0',
      install_requires=[
            'torch',
      ],
)