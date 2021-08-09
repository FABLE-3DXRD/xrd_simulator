import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xrd_simulator",
    version="0.0.0",
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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires= [ "numpy>=1.20.0",
                        "scipy",
                        "matplotlib",
                        "xfab>=0.0.4",
                        ]
)
