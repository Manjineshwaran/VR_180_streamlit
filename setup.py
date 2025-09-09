from setuptools import setup, find_packages

setup(
    name="vr180_converter",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'pyyaml',
        'numba',
    ],
    entry_points={
        'console_scripts': [
            'vr180-convert=src.main:main',
        ],
    },
)
