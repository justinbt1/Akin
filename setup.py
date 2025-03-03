from setuptools import setup

description = 'Akin is a Python library for detecting near-duplicate texts using min-hashing and locality sensitive hashing.'

readme = open('README.md', 'r')
long_description = readme.read()
readme.close()

project_urls = {
  'Homepage': 'https://github.com/justinbt1/Akin'
}

setup(
    name='akin',
    version='1.0.1',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['akin'],
    install_requires=['numpy', 'cython', 'mmh3', 'tqdm'],
    project_urls=project_urls
)
