# -*- coding: utf-8 -*-
"""

# Put this im data/coco1
Created on Sat Jun 12 15:15:09 2021

wget -c "https://pjreddie.com/media/files/train2014.zip" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/val2014.zip" --header "Referer: pjreddie.com"

# Get labels
wget -c "https://pjreddie.com/media/files/coco/labels.tgz" --header "Referer: pjreddie.com"

unzip -q train2014.zip
unzip -q val2014.zip
tar xzf labels.tgz

#put under weights
wget -c "https://pjreddie.com/media/files/darknet53.conv.74" --header "Referer: pjreddie.com"

"""


import os
import pandas as pd

def isFileAvailable(data_paths):
    
    availableFiles = []
    notFound = []
    for dp in data_paths:
        imgP = dp.split("|")[0]
        lblP = dp.split("|")[1]
    
        if os.path.exists(imgP) and os.path.exists(lblP) :
            availableFiles.append(dp)
        else: 
            notFound.append(dp)
    #print(notFound)    
    print(f"Total not found are: {len(notFound)}")
    return availableFiles
    
def get_train_val_paths(trImg_dir_path, valImg_dir_path, labels_dir_path=None):
    
    #root_path = os.getcwd()
    train_labels = os.listdir( os.path.join(labels_dir_path, "train2014"))
    val_labels = os.listdir( os.path.join(labels_dir_path, "val2014"))
    
    # Prepare training data-path file
    training_data_path = [ os.path.join(os.path.abspath(trImg_dir_path),
        x.replace("txt", "jpg")) +"|"+ os.path.join(os.path.abspath(labels_dir_path),"train2014",x)  
                          for x in train_labels]
    
    
    val_data_path = [ os.path.join(os.path.abspath(valImg_dir_path),
        x.replace("txt", "jpg")) +"|"+ os.path.join(os.path.abspath(labels_dir_path),"val2014",x) 
                          for x in val_labels]

    print(f"Total training files are: {len(training_data_path)}, checking if Training Files are missing...")
    training_data_path = isFileAvailable(training_data_path)
    
    print(f"Total validation files are : {len(val_data_path)}, checking if Testing Files are missing...")
    val_data_path  = isFileAvailable(val_data_path )
    
    return training_data_path, val_data_path


if __name__ == "__main__":
    
    labels_dir_path = "data/coco1/labels"
    trImg_dir_path = "data/coco1/train2014"
    valImg_dir_path = "data/coco1/val2014"
    
    #labels_dir_path = "labels"
    #trImg_dir_path = "train2014"
    #valImg_dir_path = "val2014"

    tr_paths, val_paths = get_train_val_paths(trImg_dir_path, valImg_dir_path, labels_dir_path)
    
    pd.DataFrame(tr_paths).to_csv("data/coco1/train_paths.txt", index=False)
    pd.DataFrame(val_paths).to_csv("data/coco1/validation_paths.txt", index=False)
