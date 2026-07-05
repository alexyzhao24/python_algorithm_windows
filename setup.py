from setuptools import setup, find_packages

setup(
    name="python_algo_production",
    version="0.1",
    packages=find_packages(),
    package_data={
        "python_algo_production.lib": ["*.dll", "*.pyd"],
    },
    include_package_data=True,
    install_requires=[
        "pytest>=8.0.0",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "pybullet>=3.2.5",
    ],
)