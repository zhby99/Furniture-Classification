# Furniture-Classification
iMaterialist Challenge (Furniture) at FGVC5, Kaggle Competition

### Pre-trained Model
For our first trial, we will use AlexNet to do transfer learning on this task.

1. Download [AlexNet weights](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy), and save the `bvlc-alexnet.npy`
 on the main repo.
2. In `alexnet.py`, we defined the feature extraction version of AlexNet, where we simple adopt the weights for all the layers before the last fully connected one, also
  we stop back-prop for previous layers during our transfer learning. We modified the output for the last
  fully connected layer to fit the number of our task's classes, which is 128.

### Pre-Processing
1. Download training data, validation data and testing data by our script, in `download.ipynb`. For example, the jpg files of training data will be stored
 in `imaterialist-challenge-furniture-2018/train_data/`.


### Training
1. After getting `X_train`, `y_train`, `X_val`, `y_val` from preprocessing. We can run `pipeline.ipynb` to train our fine-tuned model based on the training set
 and evaluate our model on the validation set.