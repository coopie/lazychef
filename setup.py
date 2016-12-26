from setuptools import setup


setup(
    name='lazychef',
    version='0.0.1',
    description='Build machine pipelines that are lazily evaluated',
    author='Sam Coope',
    author_email='sam.j.coope@gmail.com',
    url='https://github.com/coopie/lazychef',
    download_url='https://github.com/coopie/lazychef/archive/master.zip',
    license='MIT',
    install_requires=['h5py', 'numpy'],
    packages=['lazychef']
)
