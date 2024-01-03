# Read filenames in the thresholded folder
import glob
import numpy as np
import os
import cv2


def read_names(input_dir):
    '''
    Takes folder name and returns a list of filenames in the folder
    and list of 0/1 labels for each file name
    Assumes that folder contains files with name-ID-label.jpg format:
    0b02292fff587f33db7136009fa2d52c-195-0.jpg
    '''
    image_names = []
    labels = []

    for file in glob.glob(f'{input_dir}/*.jpg'):
        filename = file.split('/')[-1]
        num_colonies = int(filename.split('-')[2].strip().split('.')[0])
        if num_colonies == 0:
            labels.append(0)
        else:
            labels.append(1)
        image_names.append(filename)

    X_files = np.array(image_names) #  Array with image nbames
    y = np.array(labels)			#  Array with binary labels
    return X_files, y

from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, PrecisionRecallDisplay, precision_recall_curve
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def custom_train_val_split(X, y, val_size=0.3, random_state = 42):
    '''
    # Can not rely on the random train-val split on highly imbalaced data
    # Val subset can randomly have too few or too many positive cases
    # Instead, select pre-defined number of random images from each class and stack them together
    # Also shuffle both sets, convert labels to categorical
    # Takes: 
    # X - array of image names, y - array of labels, test_size - fraction of the set to be in val, random seed
    # Returns:
    # np arrays of image filenames: X_train, X_val and categorical arrays of labels: y_train, y_val
    '''
    np.random.seed(random_state) # For reproducibility

    #  Make two arrays with images: one for positive and one for negative images
    pos_im = [X[i] for i in range(X.shape[0]) if y[i]]
    pos_im = np.array(pos_im)
    neg_im = [X[i] for i in range(X.shape[0]) if not y[i]]
    neg_im = np.array(neg_im)
    num_pos = pos_im.shape[0]
    num_neg = neg_im.shape[0]
    print(f'Full dataset: {num_pos} images with bacteria and {num_neg} without')

    # Select random positive images
    # Make two random indeces for val and train
    idx_pos_val = np.random.choice(np.arange(num_pos), int(num_pos*val_size), replace = False)
    idx_pos_train = [i for i in np.arange(num_pos) if i not in idx_pos_val]
    #  Apply this index to the array of image names
    X_val_pos = pos_im[idx_pos_val]
    X_train_pos = pos_im[idx_pos_train]
    y_val_pos = np.ones(len(idx_pos_val))
    y_train_pos = np.ones(len(idx_pos_train))

    # Select random negative images
    # Make two random indeces for val and train
    idx_neg_val = np.random.choice(np.arange(num_neg), int(num_neg*val_size), replace = False)
    idx_neg_train = [i for i in np.arange(num_neg) if i not in idx_neg_val]
    #  Apply this index to the array of image names
    X_val_neg = neg_im[idx_neg_val]
    X_train_neg = neg_im[idx_neg_train]
    y_val_neg = np.zeros(len(idx_neg_val))
    y_train_neg = np.zeros(len(idx_neg_train))

    # Stack two classes together
    X_val = np.concatenate((X_val_neg, X_val_pos), axis = 0)
    y_val = np.concatenate((y_val_neg, y_val_pos), axis = 0)
    X_train = np.concatenate((X_train_neg, X_train_pos), axis = 0)
    y_train = np.concatenate((y_train_neg, y_train_pos), axis = 0)

    # Shuffle validation
    print("Shuffling validation dataset ....")
    num_val = X_val.shape[0]
    indices = np.arange(num_val)
    shuffled_indices = np.random.permutation(indices)
    X_val = X_val[shuffled_indices]
    y_val = y_val[shuffled_indices]
    print(f"{X_val.shape} - {y_val.shape}")

    # Shuffle training
    print("Shuffling trainig dataset ....")
    num_train = X_train.shape[0]
    indices = np.arange(num_train)
    shuffled_indices = np.random.permutation(indices)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    #Conver to int np array
    y_train = np.array(y_train).astype(int)
    y_val = np.array(y_val).astype(int)

    # Convert labeles to categorical
    num_classes = max(np.max(y_train), np.max(y_val)) + 1
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    print(f"Validation X: {X_val.shape}; y: {y_val.shape}")
    print(f"Training X: {X_train.shape}; y: {y_train.shape}")

    return X_train, X_val, y_train, y_val

def read_images(files, file_dir):
  '''
  Reads images from files located in file_dir
  Returns a 4 dimentional np array of the read images
  '''
  print(f"Reading {len(files)} images from {file_dir.split('/')[-1]}...", end = " ")
  X = []
  for file_name in files:
      file = os.path.join(file_dir, file_name)
      img = cv2.imread(file, cv2.IMREAD_COLOR)
      X.append(img)
  X = np.array(X)
  print("Done")
  return X

def read_aug_images(files, file_dir):
  '''
  Reads images from files located in file_dir
  Adds augmentation based on superposition of positive and negative images
  Returns:
  X - a 4 dimentional np array of the augmented dataset 
  y - a categorical array of labeles for augmented dataset
  '''
  print(f"Reading and agumenting {len(files)} images from {file_dir.split('/')[-1]}...", end = " ")
  # Get labels from filenames
  y = []
  for filename in files:
    num_colonies = int(filename.split('-')[2].strip().split('.')[0])
    if num_colonies == 0:
      y.append(0)
    else:
      y.append(1)

  # Sort file in to positive and negative
  pos_im_files = [files[i] for i in range(files.shape[0]) if y[i]]
  pos_im_files = np.array(pos_im_files)
  neg_im_files = [files[i] for i in range(files.shape[0]) if not y[i]]
  neg_im_files = np.array(neg_im_files)
  num_pos = pos_im_files.shape[0]
  num_neg = neg_im_files.shape[0]
  print(f'Before augmentation: {num_pos} images with bacteria and {num_neg} without')


  X = []
  y_new = []
  for lbl, file_name in zip(y, files):
      if lbl == 0:
        y_new.append(0)
        file_path = os.path.join(file_dir, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        X.append(img)
      else:
        file_path = os.path.join(file_dir, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        #Append image
        X.append(img)
        y_new.append(1)

        #Append flips
        img_flip_h = cv2.flip(img, 1)
        X.append(img_flip_h)
        y_new.append(1)
        img_flip_v = cv2.flip(img, 0)
        X.append(img_flip_v)
        y_new.append(1)

        #Append 20 superpositions of positive with random negatives
        try:
          #Get indecies of 20 random negative files
          neg_idx = np.random.choice(num_neg, 20, replace = False)
          for i, idx in enumerate(neg_idx):
              #Read the negative file
              neg_background_path= os.path.join(file_dir, neg_im_files[idx])
              neg_bckgr_im = cv2.imread(neg_background_path, cv2.IMREAD_COLOR)

              #Superimpose positive and negative
              new_im = np.maximum(img, neg_bckgr_im)

              #Append to teh ouput array
              X.append(new_im)
              y_new.append(1)
        except:
          print('Something went wrong')

  y_new = np.array(y_new).astype(int)
  X = np.array(X)

  # Shuffle resulting dataset
  print("Suffling dataset ....")
  num_im = X.shape[0]
  indices = np.arange(num_im)
  shuffled_indices = np.random.permutation(indices)
  X = X[shuffled_indices]
  y_new = y_new[shuffled_indices]
  print(f"Data set shape {X.shape}; labeles shape {y_new.shape}")

  num_pos = y_new.sum()
  num_neg = len(y_new) - y_new.sum()
  num_classes = 2

  y_new = to_categorical(y_new, num_classes=num_classes)

  print(f'After augmentation: {num_pos} images with bacteria and {num_neg} without')
  return X, y_new

def prec_rec_at_threshold(labels, pred, thresholds):
  '''
  Makes arrays of precisions and recalls given an array of thresholds
  Takes:
  labels - an array of true labels
  pred - an array of predicted probabilities
  thresholds - an array of thresholds at wich to evaluate the probabilities
  Returns:
  precisions - a list of precisions 
  recalls - a list of recalls
  '''
  precisions = []
  recalls = []
  P = sum(labels)
  for threshold in thresholds:
    TP = sum((pred > threshold) & labels)
    FP = sum((pred > threshold) & ~labels)
    precisions.append(TP/(TP+FP))
    recalls.append(TP/P)
  return precisions, recalls

from keras.callbacks import EarlyStopping
class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss',
             patience=5, verbose=0, mode='auto', start_epoch = 10, restore_best_weights=True): # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)