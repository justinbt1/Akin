from setuptools import setup

readme = open('README.md', 'r')
long_description = readme.read()
readme.close()

setup(
    name='akin',
    version='0.1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['akin'],
    install_requires=['numpy']
)
