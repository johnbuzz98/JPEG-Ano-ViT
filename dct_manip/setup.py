from setuptools import setup, Extension
from torch.utils import cpp_extension

__version__ = "0.2.3"

setup(name='dct_manip',
      ext_modules=[cpp_extension.CppExtension(
	      'dct_manip', 
	      ['dct_manip.cpp'],
		  include_dirs=['/opt/conda/envs/jpeganovit/include'],
		  library_dirs=['/opt/conda/envs/jpeganovit/lib'],
	      extra_objects=[
			'/opt/conda/envs/jpeganovit/lib/libjpeg.so',
			],
		  headers=[
			'/opt/conda/envs/jpeganovit/include/jpeglib.h',],
	      extra_compile_args=['-std=c++17']
	      ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
	  version = __version__
	  )