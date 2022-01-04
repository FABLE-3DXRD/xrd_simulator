"""Script to automatesome of the boring stuff when releasing a new version of the package to
pypi and conda. Essentially the script does the following
    1. updates the setup files with the new build version.
    2. pushes the current repo to git.
    3. builds pyi package and upload to pypi.
    4. build conda package and uploads to conda.
During the process, some user specified info is required via the command line.

This script also serves as a documentation for how to do releases.
"""

import subprocess
import os


def check_version(new_version):
    old_version = get_current_version()  # Get the old package version form setup.py
    v1 = [float(v) for v in old_version.split('.')]
    v2 = [float(v) for v in new_version.split('.')]
    if v1[0] >= v2[0] and v1[1] >= v2[1] and v1[2] >= v2[2]:
        raise ValueError(
            "Specified package version " +
            new_version +
            " must be incremented compared" +
            " to previous version " +
            old_version)


def get_current_version():
    with open("..\\setup.py", 'r') as f:
        for line in f.readlines():
            if line.strip().startswith("version"):
                return line.strip().split("=")[1].split(",")[
                    0].replace('"', '')


def bump_version(version):
    paths = ["..\\setup.py", "..\\setup.cfg"]
    patterns = [
        '    version="' +
        version +
        '",\n',
        'version = ' +
        version +
        '\n']
    for path, pattern in zip(paths, patterns):
        with open(path, 'r') as f:
            data = f.readlines()
        for i, line in enumerate(data):
            if line.strip().startswith("version"):
                data[i] = pattern
        with open(path, 'w') as f:
            f.writelines(data)


if __name__ == '__main__':

    old_version = get_current_version()
    print("Trying autorelease, current package version is " + old_version)
    new_version = sha256 = input("Enter desired new version of package: ")
    check_version(new_version)
    bump_version(new_version)

    conda_dir = os.getcwd()
    if conda_dir.split("\\")[-1] != 'conda':
        raise ValueError(
            "autorelease.py is meant to be run from within the conda directory")

    print("Commiting and pushing code to the github repository")
    os.chdir("..")
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "bumping version to " +
                   new_version + " and distributing"])
    subprocess.run(["git", "push"])

    print("Successfully bumped version")

    print("Build and upload the package to pypi")
    subprocess.run(["python", "-m", "build"])

    print(" ")
    print(" ")
    print("You need to login to your pypi account to upload, use __token__ as username")
    subprocess.run(["twine", "upload", "dist\\*" + new_version + "*"])

    os.chdir("conda")

    print(" ")
    print(" ")
    print("Package is now uploaded to pypi next go to the pypi webpage and get the build info")
    sha256 = input("Enter sha256 of tar.gz: ")
    url = input("Enter url to .tar.gz: ")

    with open("meta.yaml", 'r') as f:
        data = f.readlines()
    for i, line in enumerate(data):
        if line.strip().startswith("url"):
            data[i] = "  url: " + url + "\n"
        if line.strip().startswith("sha256"):
            data[i] = "  sha256: " + sha256 + "\n"
    with open("meta.yaml", 'w') as f:
        f.writelines(data)

    print("Building conda package")
    subprocess.run(["conda-build", "."])
    print(" ")
    print(" ")
    print("Conda releasable package has been built awaiting path to .tar.bz2 to upload")
    package = input("Enter conda build path to package (.tar.bz2 file): ")
    subprocess.run(["anaconda", "upload", "--all", package, "--skip"])
