from setuptools import setup, find_packages

setup(
    name="ghost-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-image>=0.19.0",
    ],
    author="Josh",
    description="Invariant Topological Anomaly Verification Framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
