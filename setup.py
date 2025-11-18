import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xrd_simulator",
    version="0.5",
    author="Axel Henningsson, Marc Raventós",
    author_email="nilsaxelhenningsson@gmail.com, marcraven@gmail.com",
    description="Tools for diffraction simulation of s3dxrd and powder type experiments.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/AxelHenningsson/xrd_simulator",
    project_urls={
        "Documentation": "https://axelhenningsson.github.io/xrd_simulator/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    # Keep runtime requirement open-ended: supports Python 3.9+
    python_requires=">=3.9",
    install_requires=[
        # Core numeric/scientific
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",

        # Deep learning (optional)
        "torch>=2.5.0.dev20240131",

        # Mesh generation and processing
        "meshpy",
        "meshio",

        # Visualization
        "matplotlib",
        "Pillow>=9.0.0",
        "scikit-image",

        # Scientific computing
        "numba",
        "psutil",
        "xfab",

        # I/O and serialization
        "pycifrw",
        "dill",
        "netcdf4",
        "h5py",
        "tifffile",

        # Type hints
        "typing-extensions>=4.9.0",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/nightly/cu121",
    ],
)
