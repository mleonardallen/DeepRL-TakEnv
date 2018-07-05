from setuptools import setup

setup(
  name='tak',
  version='0.1',
  description='TAK - OpenAI Gym - Reinforcement Learning Environment',
  url='https://github.com/mleonardallen/tak',
  author='Mike Allen',
  author_email='mikeleonardallen@gmail.com',
  license='MIT',
  packages=['tak'],
  install_requires=[
    'gym',
    'pygame'
  ],
  zip_safe=False
)