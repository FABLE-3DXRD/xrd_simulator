import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="xrd_simulator",
    version="0.5",
    author="Axel Henningsson",
    author_email="nilsaxelhenningsson@gmail.com, marcraven@gmail.com",
    description="Tools for diffraction simulation of s3dxrd and powder type experiments.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/AxelHenningsson/xrd_simulator",
    project_urls={
        "Documentation": "https://axelhenningsson.github.io/xrd_simulator/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "numpy",
        "meshio",
        "pygalmesh",
        "scipy",
        "numba",
        "pycifrw",
        "dill",
        "xfab",
        "netcdf4",
        "h5py",
        "torch",
        "tifffile",
    ],
)
