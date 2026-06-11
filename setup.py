from setuptools import setup, find_packages

setup(
    name="ghost-engine",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["run_benchmark"],
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "PyYAML>=6.0",
    ],
    author="Josh",
    description="Invariant Topological Anomaly Verification Framework",
    entry_points={"console_scripts": ["ghost-engine=ghost_engine.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
