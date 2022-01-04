import argparse
import subprocess
import os

def check_version(new_version):
    old_version = get_current_version() # Get the old package version form setup.py
    v1 = [float(v) for v in old_version.split('.')]
    v2 = [float(v) for v in new_version.split('.')]
    if v1[0]>=v2[0] and v1[1]>=v2[1] and v1[2]>=v2[2]:
        raise ValueError("Specified package version "+new_version+" must be incremented compared" +
            " to previous version "+ old_version)

def get_current_version():
    with open("..\setup.py", 'r') as f:
        for line in f.readlines():
            if line.strip().startswith("version"):
                return line.strip().split("=")[1].split(",")[0].replace('"','')

def bump_version( version ):
    paths = ["..\setup.py", "..\setup.cfg"]
    patterns = ['    version="'+version+'",\n', 'version = '+version+'\n']
    for path,pattern in zip(paths,patterns):
        with open(path, 'r') as f: data = f.readlines()
        for i,line in enumerate(data):
            if line.strip().startswith("version"):
                data[i] = pattern
        with open(path, 'w') as f: f.writelines( data )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version to bump the package to")
    args = parser.parse_args()
    new_version = str( args.version ) # Package version to release
    check_version( new_version )
    bump_version( new_version )

    conda_dir = os.getcwd()
    if conda_dir.split("\\")[-1]!='conda':
        raise ValueError("autorelease.py is meant to be run from within the conda directory")

    print( "Commiting and pushing code to the github repository" )
    os.chdir("..")
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "bumping version to "+new_version+" and distributing"])
    subprocess.run(["git", "push"])

    print("Successfully bumped version")
