import argparse
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version to bump the package to")
    args = parser.parse_args()

    version = str( args.version )

    with open("..\setup.cfg", 'rw') as f:
        pass # add the new version strings in

    with open("..\setup.py", 'rw') as f:
        pass # add the new version strings in

    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "bumping version to "+version+" and distributing"])
    subprocess.run(["git", "push"])

    subprocess.run(["python", "-m", "build"])

    subprocess.run(["twine", "upload", "dist\*"+version+"*"])

    subprocess.run(["conda-build", "."])
