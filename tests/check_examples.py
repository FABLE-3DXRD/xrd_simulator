import os
path = os.path.join(os.path.dirname(__file__), "../docs/source/examples/")
path = os.path.normpath(path)
for file in os.listdir(path):
    if file.endswith(".py"):
        exit = os.system("python " + os.path.join(path, file))
        if exit == 0:
            print("Found example file ", file, " run:  OK!")
        else:
            print("Found example file ", file, " run:  FAILURE!")
