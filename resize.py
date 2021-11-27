from PIL import Image
import os
import sys

paths = ["RoxImage\\train\\Background", "RoxImage\\train\\Castable", "RoxImage\\train\\Fishing", "RoxImage\\train\\Reelable",
         "RoxImage\\validation\\Background", "RoxImage\\validation\\Castable", "RoxImage\\validation\\Fishing", "RoxImage\\validation\\Reelable"]

def resize():
    for path in paths:
        items = os.listdir(path)
        for item in items:
            file_path = os.path.join(path, item)
            if os.path.isfile(file_path):
                im = Image.open(file_path)
                f, e = os.path.splitext(file_path)
                imResize = im.resize((48, 48), Image.ANTIALIAS)
                imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

def remove():
    for path in paths:
        items = os.listdir(path)
        for item in items:
            file_path = os.path.join(path, item)
            if os.path.isfile(file_path):
                if file_path.find(' resized.jpg') == -1:
                    print("Removed " + file_path)
                    os.remove(file_path)



# process here
resize()