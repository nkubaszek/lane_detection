import cv2
import os
from os import listdir
from os.path import isfile, join
from statistics import mean
import numpy as np
#gt='C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/ground_truth/'
#pred="C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/output_frames/"
from sklearn.metrics import accuracy_score, precision_score, f1_score, jaccard_score, recall_score, \
    average_precision_score, confusion_matrix

GROUNDTRUTHMASKS = "C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/ground_truth/"
PREDICTEDMASKS = "C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/output_frames/"


# Caluculate the Intersection over the UNION
def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)
def IOU(Predicted,Truth):
    true_list_new=np.array(Truth)
    pred_list_new=np.array(Predicted)
    true_list_new=true_list_new.reshape(-1)
    pred_list_new=pred_list_new.reshape(-1)
    d=precision_score(true_list_new,pred_list_new,average='binary')
    j=compute_iou(true_list_new,pred_list_new)

    print(d,j)


    return d,j



def main():
    precision_list=[]
    iou_list=[]
    recall_list=[]
    average_precison_list=[]
    # check if folders exist
    if not os.path.exists(GROUNDTRUTHMASKS) or not os.path.exists(PREDICTEDMASKS):
        os.mkdir(GROUNDTRUTHMASKS)
        os.mkdir(PREDICTEDMASKS)

    Truth_list = [f for f in listdir(GROUNDTRUTHMASKS) if isfile(join(GROUNDTRUTHMASKS, f))]
    Predicted_list = [f for f in listdir(PREDICTEDMASKS) if isfile(join(PREDICTEDMASKS, f))]

    # Check if empty folder
    if len(Truth_list) == 0 or len(Predicted_list) == 0:
        print("NO MASKS IN THE FOLDER! Terminating...")
        exit(1)


    for t in Truth_list:
        filename_list = t.split("_")
        filetested = filename_list[0]
        print(filetested)
        if os.path.isfile(PREDICTEDMASKS+filetested):
            Predicted = cv2.imread(os.path.join(PREDICTEDMASKS,filetested))/255.0
            Truth = cv2.imread(os.path.join(GROUNDTRUTHMASKS+filetested))/255.0
            precision,iou= IOU(Predicted,Truth)
            precision_list.append(precision)
            iou_list.append(iou)


    mean_precision=mean(precision_list)
    mean_iou=mean(iou_list)

    print("mean precision",mean_precision)
    print("iou",mean_iou)





if __name__ == '__main__':
    main()

