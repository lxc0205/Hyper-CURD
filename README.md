# Hyper-CURD

This is the code of Hyper-CURD. Only for personal use.

## Dependencies

- Python 3.6+
- PyTorch 0.4+
- TorchVision
- scipy

(optional for loading specific IQA Datasets)
- csv (KonIQ-10k Dataset)

## Usages

You will get a quality score ranging from 0-100, and a higher value indicates better image quality.

### Training on IQA databases

Training the baseline model on Datasets.

```
python train.py
```

Some available options:
* `--dataset`: Training dataset, support datasets: koniq-10k | live | csiq | tid2013.
* `--train_patch_num`: Sampled image patch number per training image.
* `--test_patch_num`: Sampled image patch number per testing image.
* `--batch_size`: Batch size.

When training on CSIQ dataset, please put 'csiq_label.txt' in your own CSIQ folder.

### Testing on IQA databases

Testing the baseline model on Dataset.

```
python eval.py
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--patch_num`: Number of sample patches from testing image.
* `--patch_size`: Crop size for training & testing image patches.
* `--curd`: The flag of using curd.

### curd regression

Curd regression.

```
curd/curd.cpp

regress.py
```

Some available options:
* `--dataset`: CURD training dataset, support datasets:  koniq-10k | live | csiq | tid2013.