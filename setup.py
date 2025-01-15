from setuptools import setup, find_packages

setup(
    name="vessel_segmentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'SimpleITK',
        'scipy',
        'tqdm'
    ],
) 