"""GPU port of the quantum-SK Nov3 / QAOA campaign.

Importing this package does not initialize CUDA, allocate device memory, or
change the current CUDA device. Call runtime.acquire() explicitly.
"""
__version__ = "0.1.0"
__all__ = ["__version__"]
