from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'active_imitation'))

setup(name='active_imitation',
      version='0.1-dev',
      description='An active approach to imitation learning.',
      url='na',
      author='Matt Hanczor',
      author_email='hanczor.m@gmail.com',
      license='',
)
