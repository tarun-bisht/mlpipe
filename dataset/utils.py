import pandas as pd
import os

def df_from_image_dirs(directory, image_format="jpg", 
    relative_path=False, verbose=0):
    dataframe_dict = {
        "images":[],
        "classes":[]
    }
    num_dirs = 0
    num_images = 0
    images_per_classes = []
    classes = []
    for dirs in os.listdir(directory):
        dir_path = os.path.join(directory,dirs)
        if os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if f.split(".")[1]==image_format]
            num = len(files)
            if relative_path:
                dataframe_dict["images"] = dataframe_dict["images"]+[os.path.join(dir_path,f) for f in files]
            else:
                dataframe_dict["images"] = dataframe_dict["images"]+files
            dataframe_dict["classes"] = dataframe_dict["classes"]+[dirs]*num
            num_images+=num
            images_per_classes.append(num)
            classes.append(dirs)
            num_dirs+=1
    if verbose:
        print("number of directories(classes)= ",num_dirs)
        print("total number of images= ",num_images)
        for clss, imgs in zip(classes, images_per_classes):
            print(f"{clss} : {imgs}")

    return pd.DataFrame.from_dict(dataframe_dict)