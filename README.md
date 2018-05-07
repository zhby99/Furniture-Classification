# Furniture-Classification
iMaterialist Challenge (Furniture) at FGVC5, Kaggle Competition

### Data Downloader
We have `download.py` for training data, `downloadd_valid.py` for validation data and `download_test.py` for testing data, an example running is showed below.
```bash
python download.py
```

### Data Preprocess
1. For pytorch model, use `dataset_gen.py` for making 128 directories for training set and validation set and putting the valid images into these folders according to their labels. `preprocess.py` is used to putting all valid image files into the newly created folder.
2. For tensorflow model, run `preprocess_main.py` to preprocess the images and store them in npy file. Also, download [AlexNet weights](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy), and save the `bvlc-alexnet.npy`

### Training
1. For pytorch model
```bash
python transfer_main.py
```
The weights are stored in `best_weight.pth`.
2. For tensorflow model, open `pipeline.ipynb` for training.

### Testing
```
python predict.py
```
to load the weight and make prediction on testing set, the result of prediction will be stored in `test_prediction.pth`.

### Submission File for Kaggle
```
python gen_csv.py
```
to map index of training with testing and generating csv file for submission, in `submission.csv`.
