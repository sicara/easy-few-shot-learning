from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="easyfsl",
    version="1.1.0",
    description="Ready-to-use PyTorch code to boost your way into few-shot image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sicara/easy-few-shot-learning",
    license="MIT",
    install_requires=[
        "matplotlib>=3.0.0",
        "pandas>=1.1.0",
        "torch>=1.4.0",
        "torchvision>=0.7.0",
        "tqdm>=4.1.0",
    ],
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
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)
