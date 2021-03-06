# -*- coding: utf-8 -*-
"""Improvement Unweighted Data Subsampling via Influence Function.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/117-XnyloQWTyOxkwDH8nQiV9blZXusFM

## **Basic imports**
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/Influence_Subsampling

!ls

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from inverse_hvp import inverse_hvp_lr_newtonCG
import argparse
import time
import pdb
import os
np.random.seed(0)

"""# 1. **MNIST Dataset Load and Preprocessing the data in required format**"""

# select the dataset used
dataset_name = "mnist"
# regularization parameter for Logistic Regression
C = 0.1

# tool box
def load_mnist(validation_size = 5000):
    import gzip
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder(">")
        return np.frombuffer(bytestream.read(4),dtype=dt)[0]

    def extract_images(f):
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf,dtype=np.uint8)
            data = data.reshape(num_images,rows,cols,1)
            return data
    
    def extract_labels(f):
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    data_dir = "./data"
    TRAIN_IMAGES = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
    with open(TRAIN_IMAGES,"rb") as f:
        train_images = extract_images(f)

    TRAIN_LABELS =  os.path.join(data_dir,'train-labels-idx1-ubyte.gz')
    with open(TRAIN_LABELS,"rb") as f:
        train_labels = extract_labels(f)

    TEST_IMAGES =  os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
    with open(TEST_IMAGES,"rb") as f:
        test_images = extract_images(f)

    TEST_LABELS =  os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
    with open(TEST_LABELS,"rb") as f:
        test_labels = extract_labels(f)

    # split train and val
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # preprocessing
    train_images = train_images.astype(np.float32) / 255
    test_images  = test_images.astype(np.float32) / 255
    
    # reshape for logistic regression
    train_images = np.reshape(train_images, [train_images.shape[0], -1])
    test_images = np.reshape(test_images, [test_images.shape[0], -1])
    return train_images,train_labels,test_images,test_labels


def filter_dataset(X, Y, pos_class, neg_class, mode=None):
    
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.

    They are all 10-classes image classification data sets while Logistic regression can only handle binary classification. we
    select the number 1 and 7 as positive and negative classes;
    """

    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)
    
    pos_idx = Y == pos_class
    neg_idx = Y == neg_class        
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]
    if Y.min() == -1 and mode != "svm":
        Y = (Y + 1) / 2
        Y.astype(int)
    return (X, Y)

from sklearn.model_selection import KFold # import KFold

def load_data_two_class(dataset_name,va_ratio):
  x_train,y_train,x_test,y_test = load_mnist()
  pos_class = 1
  neg_class = 7
  x_train,y_train = filter_dataset(x_train,y_train,pos_class,neg_class)
  x_va,y_va = filter_dataset(x_test,y_test,pos_class,neg_class)
  y_va = y_va.astype(int)
  y_train = y_train.astype(int)

  ######################################## 3 fold cross validation for improvement  #########################################
  kf = KFold(n_splits=3) # Define the split - into 3 folds 
  kf.get_n_splits(x_train) # returns the number of splitting iterations in the cross-validator
  print(kf) 
  KFold(n_splits=3, random_state=None, shuffle=False)
  i=0

  for train_index, test_index in kf.split(x_train):
    if (i==0):
      x_train_0 = x_train[train_index[0]:train_index[-1]]
      y_train_0 = y_train[train_index[0]:train_index[-1]]
      x_val_0 = x_train[test_index[0]:test_index[-1]]
      y_val_0 = y_train[test_index[0]:test_index[-1]]

    elif (i==1):
      x_train_1 = x_train[train_index[0]:train_index[-1]]
      y_train_1 = y_train[train_index[0]:train_index[-1]]
      x_val_1 = x_train[test_index[0]:test_index[-1]]
      y_val_1 = y_train[test_index[0]:test_index[-1]]

    elif (i==2):
      x_train_2 = x_train[train_index[0]:train_index[-1]]
      y_train_2 = y_train[train_index[0]:train_index[-1]]
      x_val_2 = x_train[test_index[0]:test_index[-1]]
      y_val_2 = y_train[test_index[0]:test_index[-1]]

    i=i+1
    
  ############################################################################################################################
  
  x_te = x_va
  y_te = y_va

  return x_train_0,y_train_0,x_val_0,y_val_0,x_train_1,y_train_1,x_val_1,y_val_1,x_train_2,y_train_2,x_val_2,y_val_2,x_te,y_te

# load data, pick 3 cross validation as the Va set
x_train_0,y_train_0,x_va_0,y_va_0,x_train_1,y_train_1,x_va_1,y_va_1,x_train_2,y_train_2,x_va_2,y_va_2,x_te,y_te = load_data_two_class(dataset_name,va_ratio=0.3)


print("x_te,    nr sample {}, nr feature {}".format(x_te.shape[0],x_te.shape[1]))
print("Te: Pos {} Neg {}".format(y_te[y_te==1].shape[0],y_te[y_te==0].shape[0]))

print(" for k fold 0 -------------------->")
print("x_train, nr sample {}, nr feature {}".format(x_train_0.shape[0],x_train_0.shape[1]))
print("x_va,    nr sample {}, nr feature {}".format(x_va_0.shape[0],x_va_0.shape[1]))
print("Tr: Pos {} Neg {}".format(y_train_0[y_train_0==1].shape[0],y_train_0[y_train_0==0].shape[0]))
print("Va: Pos {} Neg {}".format(y_va_0[y_va_0==1].shape[0],y_va_0[y_va_0==0].shape[0]))

print(" for k fold 1 -------------------->")
print("x_train, nr sample {}, nr feature {}".format(x_train_1.shape[0],x_train_1.shape[1]))
print("x_va,    nr sample {}, nr feature {}".format(x_va_1.shape[0],x_va_1.shape[1]))
print("Tr: Pos {} Neg {}".format(y_train_1[y_train_1==1].shape[0],y_train_1[y_train_1==0].shape[0]))
print("Va: Pos {} Neg {}".format(y_va_1[y_va_1==1].shape[0],y_va_1[y_va_1==0].shape[0]))

print(" for k fold 2 -------------------->")
print("x_train, nr sample {}, nr feature {}".format(x_train_2.shape[0],x_train_2.shape[1]))
print("x_va,    nr sample {}, nr feature {}".format(x_va_2.shape[0],x_va_2.shape[1]))
print("Tr: Pos {} Neg {}".format(y_train_2[y_train_2==1].shape[0],y_train_2[y_train_2==0].shape[0]))
print("Va: Pos {} Neg {}".format(y_va_2[y_va_2==1].shape[0],y_va_2[y_va_2==0].shape[0]))

"""# 2. **Flip labels for some of the images to make the data noisy so that we can show how our method discrad these kind of noisy data hence flipping the labels of some data in below code**"""

# get the subset samples number
num_tr_sample = x_train_0.shape[0]
sample_ratio = 0.6
obj_sample_size = int(sample_ratio * num_tr_sample)

"""
Our unweighted method can downweight the bad cases which cause high test loss to the our model, which is an important reason of its ability to improve result with less data.
To show the performance of our methods in noisy label situation, we perform addtional experiments with some training
labels being flipped. The results show the enlarging superiority of our subsampling methods 
"""

# flip labels
idxs = np.arange(y_train_0.shape[0])
flip_ratio = 0.4
np.random.shuffle(idxs)
num_flip = int(flip_ratio * len(idxs))
y_train_0[idxs[:num_flip]] = np.logical_xor(np.ones(num_flip), y_train_0[idxs[:num_flip]]).astype(int)

idxs = np.arange(y_train_1.shape[0])
np.random.shuffle(idxs)
num_flip = int(flip_ratio * len(idxs))
y_train_1[idxs[:num_flip]] = np.logical_xor(np.ones(num_flip), y_train_1[idxs[:num_flip]]).astype(int)

idxs = np.arange(y_train_2.shape[0])
flip_ratio = 0.4
np.random.shuffle(idxs)
num_flip = int(flip_ratio * len(idxs))
y_train_2[idxs[:num_flip]] = np.logical_xor(np.ones(num_flip), y_train_2[idxs[:num_flip]]).astype(int)

"""# 3. **Training model on the full data set (ˆθ on the full Tr)**"""

# define the full-set-model 

""" 
model : logistic regression:
C: Inverse of regularization strength. smaller values specify stronger regularization.
fit_intercept: constant (a.k.a. bias or intercept) should be added to the decision function.
tol: Tolerance for stopping criteria.
solver: Algorithm to use in the optimization problem.For small datasets, ‘liblinear’ is a good choice
multi_class: ‘ovr’, then a binary problem is fit for each label.
max_iterint: Maximum number of iterations taken for the solvers to converge
"""

clf = LogisticRegression(
        C = C,
        fit_intercept=False,
        tol = 1e-8,
        solver="liblinear",
        multi_class="ovr",
        max_iter=100,
        warm_start=False,
        verbose=1,
        )

clf.fit(x_train_0,y_train_0)

# on Va

y_va_pred_0 = clf.predict_proba(x_va_0)[:,1] 
#predict_proba : Probability estimates. The returned estimates for all classes are ordered by the label of classes.
full_logloss_0 = log_loss(y_va_0,y_va_pred_0) 
#Log loss, aka logistic loss or cross-entropy loss. This is the loss function defined as the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data y_true. 
#For a single sample with true label yt in {0,1} and estimated probability yp that yt = 1, the log loss is
#-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
weight_ar = clf.coef_.flatten() 
#Coefficient of the features in the decision function. coef_ is of shape (1, n_features) when the given problem is binary.


# on Va
clf.fit(x_train_1,y_train_1)
y_va_pred_1 = clf.predict_proba(x_va_1)[:,1] 
full_logloss_1 = log_loss(y_va_1,y_va_pred_1) 
weight_ar = clf.coef_.flatten() 

# on Va
clf.fit(x_train_2,y_train_2)
y_va_pred_2 = clf.predict_proba(x_va_2)[:,1] 
full_logloss_2 = log_loss(y_va_2,y_va_pred_2) 
weight_ar = clf.coef_.flatten() 

# on Te

y_te_pred = clf.predict_proba(x_te)[:,1]
full_te_logloss = log_loss(y_te,y_te_pred)
full_te_auc = roc_auc_score(y_te, y_te_pred)
#The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
# The true-positive rate is also known as sensitivity, recall or probability of detection
# The false-positive rate is also known as probability of false alarm 
# AUC measures how true positive rate (recall) and false positive rate trade off

y_te_pred = clf.predict(x_te)
full_te_acc = (y_te == y_te_pred).sum() / y_te.shape[0]


# print full-set-model results
print("[FullSet] Va 0 logloss {:.6f}".format(full_logloss_0))
print("[FullSet] Va 1 logloss {:.6f}".format(full_logloss_1))
print("[FullSet] Va 2 logloss {:.6f}".format(full_logloss_2))
print("[FullSet] Te logloss {:.6f}".format(full_te_logloss))

"""# 4. **compute the influence function (IF) for each sample in training set**"""

def grad_logloss_theta_lr(label,ypred,x,C=0.03,has_l2=True,scale_factor=1.0):
    """Return d l_i / d_theta = d l_i / d_ypred * d y_pred / d theta
        grad_logloss_theta: gradient on the theta, shape: [n,]
    """
    # The isinstance() function returns True if the specified object is of the specified type, otherwise False.
    if not isinstance(label,np.ndarray) or not isinstance(ypred,np.ndarray):
        label = np.array(label).flatten()
        ypred = np.array(ypred).flatten()


    grad_logloss_theta = C * x.T.dot(ypred-label)

    return scale_factor * grad_logloss_theta

def batch_grad_logloss_lr(label,ypred,x,C=0.03,scale_factor=1.0):
    """Return gradient on a batch.
        batch_grad: gradient of each sample on parameters,
            has shape [None,n]
    """
    diffs = ypred - label
    if isinstance(x,np.ndarray):
        diffs = diffs.reshape(-1,1)
        batch_grad = x * diffs
    else:
        diffs = sparse.diags(diffs)
        batch_grad = x.T.dot(diffs).T
    batch_grad = sparse.csr_matrix(C * batch_grad)      
    return scale_factor * batch_grad

# building precoditioner
test_grad_loss_val = grad_logloss_theta_lr(y_va_0,y_va_pred_0,x_va_0,C,0.1/(num_tr_sample*C))  #Return d l_i / d_theta

tr_pred_0 = clf.predict_proba(x_train_0)[:,1]
batch_size = 10000
M = None
total_batch = int(np.ceil(num_tr_sample / float(batch_size)))

for idx in range(total_batch):
    batch_tr_grad = batch_grad_logloss_lr(y_train_0[idx*batch_size:(idx+1)*batch_size],
        tr_pred_0[idx*batch_size:(idx+1)*batch_size],
        x_train_0[idx*batch_size:(idx+1)*batch_size],
        C,
        1.0)

    sum_grad = batch_tr_grad.multiply(x_train_0[idx*batch_size:(idx+1)*batch_size]).sum(0)
    if M is None:
        M = sum_grad
    else:
        M = M + sum_grad       
M = M + 0.1/(num_tr_sample*C) * np.ones(x_train_0.shape[1])
M = np.array(M).flatten()

# computing the inverse Hessian-vector-product
#The Hessian Matrix is a square matrix of second ordered partial derivatives of a scalar function.
#It is of immense use in linear algebra as well as for determining points of local maxima or minima
iv_hvp = inverse_hvp_lr_newtonCG(x_train_0,y_train_0,tr_pred_0,test_grad_loss_val,C,True,1e-5,True,M,0.1/(num_tr_sample*C))

# get influence score
total_batch = int(np.ceil(x_train_0.shape[0] / float(batch_size)))
predicted_loss_diff = []
for idx in range(total_batch):
    train_grad_loss_val = batch_grad_logloss_lr(y_train_0[idx*batch_size:(idx+1)*batch_size],
        tr_pred_0[idx*batch_size:(idx+1)*batch_size],
        x_train_0[idx*batch_size:(idx+1)*batch_size],
        C,
        1.0)
    predicted_loss_diff.extend(np.array(train_grad_loss_val.dot(iv_hvp)).flatten())    
predicted_loss_diffs = np.asarray(predicted_loss_diff)

print("=="*30)
print("IF(inpulance function) Stats: mean {:.10f}, max {:.10f}, min {:.10f}".format(
    predicted_loss_diffs.mean(), predicted_loss_diffs.max(), predicted_loss_diffs.min())
)

"""# **5.compute the sampling probability of each sample in training set**"""

def select_from_one_class(y_train,prob_pi,label,ratio):
    # select positive and negative samples respectively
    num_sample = y_train[y_train==label].shape[0]
    all_idx = np.arange(y_train.shape[0])[y_train==label]
    label_prob_pi = prob_pi[all_idx]
    obj_sample_size = int(ratio * num_sample)

    sb_idx = None
    iteration = 0
    while True:
        rand_prob = np.random.rand(num_sample)
        iter_idx = all_idx[rand_prob < label_prob_pi]
        if sb_idx is None:
            sb_idx = iter_idx
        else:
            new_idx = np.setdiff1d(iter_idx, sb_idx)
            diff_size = obj_sample_size - sb_idx.shape[0]
            if new_idx.shape[0] < diff_size:
                sb_idx = np.union1d(iter_idx, sb_idx)
            else:
                new_idx = np.random.choice(new_idx, diff_size, replace=False)
                sb_idx = np.union1d(sb_idx, new_idx)
        iteration += 1
        if sb_idx.shape[0] >= obj_sample_size:
            sb_idx = np.random.choice(sb_idx,obj_sample_size,replace=False)
            return sb_idx

        if iteration > 100:
            diff_size = obj_sample_size - sb_idx.shape[0]
            leave_idx = np.setdiff1d(all_idx, sb_idx)
            # left samples are sorted by their IF
            # leave_idx = leave_idx[np.argsort(prob_pi[leave_idx])[-diff_size:]]
            leave_idx = np.random.choice(leave_idx,diff_size,replace=False)
            sb_idx = np.union1d(sb_idx, leave_idx)
            return sb_idx

# build sampling probability

sigmoid_k = 10  # parameter for the sigmoid sampling function
phi_ar = - predicted_loss_diffs
IF_interval = phi_ar.max() - phi_ar.min()
a_param = sigmoid_k / IF_interval
prob_pi = 1 / (1 + np.exp(a_param * phi_ar))
print("Pi Stats:",np.percentile(prob_pi,[10,25,50,75,90]))

"""# **6. Do subsampling**"""

def load_data_two_class(dataset_name,va_ratio):
  x_train,y_train,x_test,y_test = load_mnist()
  pos_class = 1
  neg_class = 7
  x_train,y_train = filter_dataset(x_train,y_train,pos_class,neg_class)
  x_va,y_va = filter_dataset(x_test,y_test,pos_class,neg_class)
  y_va = y_va.astype(int)
  y_train = y_train.astype(int)

  num_va_sample = int((1-va_ratio) * x_train.shape[0])
  x_val = x_train[num_va_sample:]
  y_val = y_train[num_va_sample:]
  x_train = x_train[:num_va_sample]
  y_train = y_train[:num_va_sample]
  x_te = x_va
  y_te = y_va

  return x_train,y_train,x_val,y_val,x_te,y_te

# load data, pick 30% as the Va set
x_train,y_train,x_va,y_va,x_te,y_te = load_data_two_class(dataset_name,va_ratio=0.3)


pos_idx = select_from_one_class(y_train_0,prob_pi,1,sample_ratio)
neg_idx = select_from_one_class(y_train_0,prob_pi,0,sample_ratio)
sb_idx = np.union1d(pos_idx,neg_idx)
sb_x_train = x_train[sb_idx]
sb_y_train = y_train[sb_idx]

"""# **7. train a subset-model on the reduced data set**"""

clf.fit(sb_x_train,sb_y_train)
y_va_pred = clf.predict_proba(x_va_2)[:,1]
sb_logloss = log_loss(y_va_2, y_va_pred)
sb_weight = clf.coef_.flatten()
diff_w_norm = np.linalg.norm(weight_ar - sb_weight)
sb_size = sb_x_train.shape[0]
y_te_pred = clf.predict_proba(x_te)[:,1]
sb_te_logloss = log_loss(y_te,y_te_pred)
sb_te_auc = roc_auc_score(y_te, y_te_pred)
y_te_pred = clf.predict(x_te)
sb_te_acc = (y_te == y_te_pred).sum() / y_te.shape[0]

"""## **Result for comparision**

## **Comparision of test accuracy on our UIDS technique with k cross  , before implementation without k cross**
"""

print("=="*30)
print("Result Summary on Te (ACC and AUC)")
print("[UIDS_k_fold]  acc {:.6f}, auc {:.6f} # {}".format(sb_te_acc,sb_te_auc, sb_size))
print("[UIDS_before]   acc {:.6f}, auc {:.6f} # {}".format(0.984281, 0.998802, 4994)) #taken from previous experiment where we do normal implementation of data without anf k cross validation
print("=="*30)