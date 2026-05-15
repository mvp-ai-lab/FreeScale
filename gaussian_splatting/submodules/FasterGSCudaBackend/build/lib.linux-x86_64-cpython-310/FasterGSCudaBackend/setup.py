import os
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'A refactored CUDA implementation of the 3DGS rasterizer.'

ENABLE_FASTMATH = True
ENABLE_NVCC_LINEINFO = False

setup_dir = Path(__file__).parent.absolute()
extension_name = setup_dir.name

# Explicitly list all source files
sources = [
    'torch_bindings/bindings.cpp',
    'adam/src/adam.cu',  # Add explicit paths
    # Add other source files here
]

# Or use glob to find all source files
sources = []
sources.extend(glob('torch_bindings/*.cpp'))
sources.extend(glob('adam/src/**/*.cu', recursive=True))
sources.extend(glob('adam/src/**/*.cpp', recursive=True))

# Explicitly set include directories with absolute paths for debugging
include_dirs = [
    os.path.join(setup_dir, 'utils'),
    os.path.join(setup_dir, 'adam/include'),
    os.path.join(setup_dir, 'adam'),
    setup_dir,  # The root directory
]

# Convert to strings (setuptools expects strings, not Path objects)
include_dirs = [str(d) for d in include_dirs]
sources = [str(Path(s).absolute()) for s in sources]  # Use absolute paths for sources

print("Include dirs:", include_dirs)
print("Sources:", sources)

cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17']
if ENABLE_FASTMATH:
    cxx_flags.append('-O3')
    nvcc_flags.append('-O3')
    nvcc_flags.append('-use_fast_math')
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }
)

setup(
    name=extension_name,
    author=__author__,
    packages=[extension_name, f'{extension_name}.torch_bindings'],
    package_dir={
        extension_name: '.',
        f'{extension_name}.torch_bindings': 'torch_bindings',
    },
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)