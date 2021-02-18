from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="easy-fsl",
    version="0.1.0",
    description="Ready-to-use PyTorch code to boost your way into few-shot image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sicara/tuto-fsl",
    license="MIT",
    install_requires=["click>=6.7", "numpy>=1.10", "tensorflow-addons>=0.9.1"],
    extras_require={"publish": ["bumpversion>=0.5.3", "twine>=1.13.0"]},
    packages=find_packages(),
    python_requires=">=3.6",
    entry_points={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)