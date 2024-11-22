# setup.py
from setuptools import setup, find_packages

setup(
    name="vessel_segmentation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'nibabel',
        'networkx',
        'scikit-image',
        'matplotlib',
        'vtk',
        'pyyaml',
        'seaborn'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for vessel segmentation in medical images",
    keywords="medical-imaging vessel-segmentation image-processing",
    python_requires='>=3.7'
)