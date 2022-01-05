# custom bash script to build the docs and put them in a place that is
# visible to github pages, which unfortunately cannot see into the
# build folder.
import os
import shutil

out = os.system("make html")
if out != 0:
    print("")
    print("")
    raise ValueError("Failed to build docs")
for file in os.listdir(os.path.join("build","html")):
    if not file.startswith('_'):
        shutil.copy2(os.path.join(os.path.join("build", "html"), file), ".")
print("Copied all docs to path visible by github pages")
