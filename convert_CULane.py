# In[3]:


from shutil import copy
from shutil import copyfile
import os
import sys
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from os import walk
import IPython
#IPython.embed() # to debug in notebook
import random

from PIL.Image import Image

print("Successfully imported all")


# ## Constants and paths

# In[4]:


cwd ="C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/"



# In[5]:


display_results = False if sys.argv[-1] == "--no-vis" else True


# In[6]:

#Tworzenie nowych folderów
export_folder_name ="C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/culane_dataset_lanenet/"
print("Export folder: ", export_folder_name)

export_img_dir = export_folder_name + "image/"
export_instance_dir = export_folder_name + "gt_image_instance/"
export_binary_dir = export_folder_name + "gt_image_binary/"

# #Tworzenie nowych folderów
base_export_dir = "C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/culane_dataset_lanenet/"
print("base_export_dir: ", base_export_dir)

image_dir = base_export_dir + "image/"
instance_dir = base_export_dir + "gt_image_instance/"
binary_dir = base_export_dir + "gt_image_binary/"
validation_path = base_export_dir + "val.txt"
train_path = base_export_dir + "train.txt"

data_base_dir = "C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/CULane/CULane/"
print("CULane path: ", data_base_dir)

# Lokalizacja adnotacji
annotation_dir = data_base_dir + "laneseg_label_w16/laneseg_label_w16"

# Take all images from annotation directory, get corresponding actual image from CULane,
# convert the image into annotations with values 20, 70, 120, 170
# convert the image to all white annotations


# ## Convert notebook to python file

# In[7]:


#!jupyter nbconvert --to script CULane_Remake.ipynb
# Can use the command line argument instead:
#$jupyter nbconvert --to script CULane_Remake.ipynb


# ## Print pretty with colors

# In[8]:


# Credit: https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
class CMD_C:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'   # End formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ## Helper functions

# In[9]:


def make_dir(dir_path):
    if (os.path.exists(dir_path)):
        print(dir_path, CMD_C.OKGREEN, " already exists in the current working direcrectory: ", CMD_C.ENDC, cwd, sep="")
    else:
        try:
            os.mkdir(dir_path)
        except OSError:
            #print(FAIL, "Could not create destination folder: ", dir_path, ENDC)
            return False
        else:
            print("Sucessfully made destination folder: ", dir_path)
    return True


# # Gather all file paths

# In[10]:


# Credit: https://www.mkyong.com/python/python-how-to-list-all-files-in-a-directory/
annotation_paths = []
for r, d, files in os.walk(annotation_dir):
    for file in files:
        annotation_paths.append(os.path.join(r, file).replace("\\","/"))
print("Number of annotations total: ", CMD_C.OKBLUE, len(annotation_paths), CMD_C.ENDC, sep="")


# # Gather corresponding actual pictures

# In[12]:


paths = []                                                                    # CULane img, CULane annot, In LaneNet img, in LaneNet binary, in LaneNet inst, copy to img, copy to binary, copy to inst
new_file_name_counter = 0                                             # Since CULane has many files with the same name in different directories, enumerate the pictures.
num_blank = 0

#sprawdzenie, czy nie ma obrazów pustych
print("Gathering paths and determining which images are blank...")
percent = len(annotation_paths) // 100
print_at = percent
#iteracja po obrazach
for i, annot_path in enumerate(annotation_paths):
    # Pominięcie obrazów pustych
    img = cv2.imread(annot_path, cv2.IMREAD_COLOR)
    if np.sum(img) == 0:
        num_blank += 1
        continue
    tok = annot_path.split("/")
    filename = tok[-1]
    filename_no_ext = filename.split(".")[0]
    path = "/".join(tok[-3:-1])
    path2 = "/".join(tok[-3:-2])
    #ścieżka do obrazów
    path_to_file = data_base_dir +path2 + "/" + path + "/" + filename_no_ext + ".jpg"
    #print(path_to_file)
    # Check to see if that file actually exists, because not all annotated pictures have pitures?
    # Or I deleted some pictures?
    # Should /CULane_Dataset/driver_161_90frame/06031716_0888.MP4/ be empty?

    #zapisanie obrazów z rozszerzeniem .png
    image_path_lanenet = image_dir + str(new_file_name_counter) + ".png"
    binary_path_lanenet = binary_dir + str(new_file_name_counter) + ".png"
    instance_path_lanenet = instance_dir + str(new_file_name_counter) + ".png"

    export_image_path_lanenet = export_img_dir + str(new_file_name_counter) + ".png"
    export_binary_path_lanenet = export_binary_dir + str(new_file_name_counter) + ".png"
    export_instance_path_lanenet = export_instance_dir + str(new_file_name_counter) + ".png"
    paths.append((path_to_file, annot_path, image_path_lanenet, binary_path_lanenet, instance_path_lanenet, export_image_path_lanenet, export_binary_path_lanenet, export_instance_path_lanenet))

    new_file_name_counter += 1

    if i >= print_at:
        print(str(print_at) + "/" + str(len(annotation_paths)), "so far, num blanks = ", str(num_blank))
        print_at += percent

print("Number blank annotations: ", num_blank)
"""
print("Example paths")
for i in range(0, 2):
    print(paths[i][0])
    print(paths[i][1])
    print(paths[i][2])
    print(paths[i][3])
    print(paths[i][4])
    print(paths[i][5])
    print(paths[i][6])
    print(paths[i][7], "\n")
"""

# # Make destination folder to save all outputs to

# In[ ]:
"""

folders_to_create = [export_folder_name, export_folder_name + "image/", export_folder_name + "gt_image_instance/", export_folder_name + "gt_image_binary/"]
for folder in folders_to_create:
    if make_dir(folder) == False:
        print(CMD_C.FAIL, "COULD NOT CREATE THE FOLDER: ", folder, CMD_C.ENDC, sep="")

"""
# # Save train.txt and val.txt

# In[ ]:


# Podział na zbiór treningowy i walidacyjny oraz zapisanie ścieżek do pliku tekstowego
with open(export_folder_name + "train.txt", "w") as train_file:
    with open(export_folder_name + "val.txt", "w") as val_file:
        for _, _, img_path, bin_path, inst_path, _, _, _ in paths:
            str_to_write = img_path + " " + bin_path + " " + inst_path + "\n"
            if random.random() < 0.9:
                train_file.write(str_to_write)
            else:
                val_file.write(str_to_write)
print("Wrote train.txt and val.txt")


# ## Main loop

# In[ ]:


plt.rcParams["figure.figsize"] = (20,10)
#r_ind = 43456  # A random curvy section
file_no = 0
for CU_img_path, CU_annot_path, _, _, _, export_img_path, export_bin_path, export_inst_path in paths:
    file_no += 1
    print("Processing image ", CMD_C.OKGREEN, file_no, CMD_C.ENDC, " / ", len(paths), " : ", sep="", end="")
    print(CMD_C.OKGREEN,  "{0:.1f}".format(100 * file_no / len(paths)), "% | ", CMD_C.ENDC, sep="", end="")
    # Skopiowanie obrazu do nowego folderu
    try:
        # Sprawdzenie, czy obraz nie jest nadpisywany
        if os.path.isfile(export_img_path) == True:
            print(CMD_C.FAIL, export_img_path, " already exists, DO NOT WANT TO OVERWRITE IT SO SKIPPING", CMD_C.ENDC)
            continue
        img = cv2.imread(CU_img_path, cv2.IMREAD_COLOR)
        save_success = cv2.imwrite(export_img_path, img)
        if save_success == False:
            print(CMD_C.FAIL, "Could not copy file: ", CU_img_path, CMD_C.ENDC, " to ", CMD_C.FAIL, export_img_path, CMD_C.ENDC, sep="", end="")
    except:
        print(CMD_C.FAIL, "Could not copy file: ", CU_img_path, CMD_C.ENDC, " to ", CMD_C.FAIL, export_img_path, CMD_C.ENDC, sep="", end="")
    else:
        print(" copied img | ", sep="", end="")


    # Wczytanie zaadnotowanego obrazu ze zbioru
    try:
        img = cv2.imread(CU_annot_path, cv2.IMREAD_GRAYSCALE)
        # Konwersja do obrazu binarnego
        ret, white = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        #Zmiana skali
        change_scale = img.copy()
        change_to = {1:20, 2:70, 3:120, 4:170}

        for old, new in change_to.items():
            change_scale[change_scale == old] = new

    except Exception as ex:
        print("Error", ex)
    else:
        pass
   
    # Zapis ground truth
    save_success = cv2.imwrite(export_bin_path, white)
    if save_success:
        print("binary saved | ", sep="", end = "")
    else:
        print(CMD_C.FAIL, "FAILED TO SAVE IMAGE", CMD_C.ENDC, export_bin_path)


    save_success = cv2.imwrite(export_inst_path, change_scale)
    if save_success:
        print("annotation saved | ", sep="", end = "")
    else:
        print(CMD_C.FAIL, "FAILED TO SAVE IMAGE", CMD_C.ENDC, export_inst_path)



    print()

# In[ ]:


print("All done!")
