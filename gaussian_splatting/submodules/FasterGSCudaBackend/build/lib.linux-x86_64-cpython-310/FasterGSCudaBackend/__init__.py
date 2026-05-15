from pathlib import Path

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',
]

try:
    # Note: The import path changed
    from FasterGSCudaBackend.torch_bindings import FusedAdam
    __all__ = ['FusedAdam']
except ImportError as e:
    print(f"Debug: Available modules: {dir()}")
    print(f"Debug: Error details: {e}")
    raise ImportError(f"Failed to import {__extension_name__}. Install with: {' '.join(__install_command__)}") from e