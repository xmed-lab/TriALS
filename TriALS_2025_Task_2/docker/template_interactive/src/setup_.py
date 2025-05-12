from setuptools import setup, find_namespace_packages

setup(name='nnunetv2',
      description='Sliding Window-based Interactive Segmentation of Volumetric Medical Images.',
      url='https://github.com/Zrrr1997/SW-FastEdit',
      author='Zdravko Marinov, Matthias Hadlich',
      author_email='zdravko.marinov@kit.edu',
      license='Apache License Version 2.0, January 2004',
      python_requires=">=3.9",
      install_requires=[
        cupy-cuda12x
        cucim
        pynvml
      ],
      )
