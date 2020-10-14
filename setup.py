from setuptools import setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="featureselect",
    version="0.0.1",
    description="Say hello!",
    py_modules=["featureselect"],
    package_dir={"": "src"},
    long_description=long_description,
    # long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/himanshu-dutta/featureselect",
    author="Himanshu Dutta",
    author_email="meet.himanshu.dutta@gmail.com",
    install_requires=["numpy >= 1.9.0", "scikit-learn >= 0.23.0"],
)