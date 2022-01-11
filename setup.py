import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="xrd_simulator",
    version="0.3.5",
    author="Axel Henningsson",
    author_email="nilsaxelhenningsson@gmail.com",
    description="Tools for diffraction simulation of s3dxrd type experiments.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/AxelHenningsson/xrd_simulator",
    project_urls={
        "Documentation": "https://axelhenningsson.github.io/xrd_simulator/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8,<3.9",
    install_requires=["matplotlib==3.3.1",
                      "numpy==1.21.4",
                      "meshio==5.0.2",
                      "pygalmesh==0.10.6",
                      "scipy==1.6.1",
                      "numba==0.53.0",
                      "pycifrw==4.4.3"]
)
