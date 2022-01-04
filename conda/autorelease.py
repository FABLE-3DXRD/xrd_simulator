
import argparse
import subprocess
import os

def get_current_version():
    with open("..\setup.py", 'r') as f:
        for line in f.readlines():
            if line.strip().startswith("version"):
                return line.strip().split("=")[1].split(",")[0].replace('"','')

if __name__ == '__main__':
    version = get_current_version()

    print( "Build and upload the package to pypi" )
    os.chdir("..")
    subprocess.run(["python", "-m", "build"])
    subprocess.run(["twine", "upload", "dist\*"+version+"*"])

    print( "Building and uploading conda package" )
    os.chdir("conda")
    subprocess.run(["conda-build", "."])
    path = r'C:\Users\Henningsson\anaconda3\conda-bld\xrd_simulator-*'+version+r'*.tar.bz2'
    subprocess.run(["anaconda", "upload", "--all", path, "--skip"])