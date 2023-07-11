from setuptools import setup, find_packages

PROJECTS = {
    "aistron.projects.aisformer": "projects/AISFormer/aisformer/",
    "aistron.projects.bcnet": "projects/BCNet/bcnet/",
}
setup(name='aistron',
      version='0.1.0',
      author='Minh Tran',
      packages=find_packages() + list(PROJECTS.keys()),
      package_dir=PROJECTS,
      description='An AIS library',
      license='Apache License 2.0',
      install_requires=[
            'torch',
      ],
)