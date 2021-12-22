import os
path = os.path.join(os.path.dirname(__file__), "../docs/source/examples/")
path = os.path.normpath(path)
for file in os.listdir(path):    
    if file.endswith(".py"):
        try:
            os.system("python "+ os.path.join( path, file ))
            print("Found example file ", file, " OK!")  
        except: 
            print("Found example file ", file, " FAILURE!")