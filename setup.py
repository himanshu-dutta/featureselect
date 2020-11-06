from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="featureselect",
    keywords="featureselect",
    version="0.0.5",
    description="An elegant and effectice solution to get best set of features from a numerical dataset!",
    py_modules=["featureselect"],
    packages=find_packages(),
    # package_dir={"": "featureselect", "base": "featureselect/base"},
    long_description=long_description,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
    ],
    url="https://github.com/himanshu-dutta/featureselect",
    author="Himanshu Dutta",
    author_email="meet.himanshu.dutta@gmail.com",
    install_requires=["numpy >= 1.9.0", "scikit-learn >= 0.23.0"],
)