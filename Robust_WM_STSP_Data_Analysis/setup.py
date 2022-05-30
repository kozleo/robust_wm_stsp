import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

setuptools.setup(
    name="stabda",
    version="0.0.1",
    description="Code for experiments in Kozachkov et al. 2022",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="John Tauber",
    author_email="jtauber@mit.edu",
    license="LICENSE",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "pandas",
        "xarray",
    ],
    classifiers=[
        "Intended Audience :: Me",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    zip_safe=False,
)
