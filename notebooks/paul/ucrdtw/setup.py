from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension('_ucrdtw', ['src/_ucrdtw.c', 'src/ucrdtw.c'])

setup(
    name='ucrdtw',
    #packages = ['ucrdtw'],
    version='0.202',
    url='https://pypi.python.org/pypi/pip',
    maintainer='m3at',
    maintainer_email='paul@alpacadb.com',
    description='Fast Dynamic Time Warping, O(n log n) ',
    long_description='Based on the following paper: http://www.cs.ucr.edu/~eamonn/SIGKDD_trillion.pdf',
    license='MIT',
    keywords='Dynamic Time Wraping',
    #install_requires=['numpy'],
    ext_modules=[c_ext],
    include_dirs=[numpy.get_include()],
)
