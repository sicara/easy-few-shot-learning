from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="easyfsl",
    version="0.1.0",
    description="Ready-to-use PyTorch code to boost your way into few-shot image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sicara/easy-few-shot-learning",
    license="MIT",
    install_requires=[
        "loguru>=0.5.3",
        "matplotlib>=3.3.4",
        "pandas>=1.2.1",
        "torch>=1.7.1",
        "torchvision>=0.8.2",
        "tqdm>=4.56.0",
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
        "Operating System :: OS Independent",
    ],
)
