from setuptools import setup

setup(
  name='tak_env',
  version='0.1',
  description='TAK - OpenAI Gym - Reinforcement Learning Environment',
  url='https://github.com/mleonardallen/tak_env',
  author='Mike Allen',
  author_email='mikeleonardallen@gmail.com',
  license='GPL',
  packages=['tak_env'],
  install_requires=[
    'gym',
    'pygame',
    'numpy',
    'networkx==2.1',
    'memoization',
  ],
  zip_safe=False,
  include_package_data=True
)