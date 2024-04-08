# Hyper-CURD

This is the code of Hyper-CURD. Only for personal use.

## Dependencies

- Python 3.6+
- PyTorch 0.4+
- TorchVision
- scipy

(optional for loading specific IQA Datasets)
- csv (KonIQ-10k Dataset)
- openpyxl (BID Dataset)

## Usages

You will get a quality score ranging from 0-100, and a higher value indicates better image quality.

### Training on IQA databases

Training the baseline model on Datasets.

```
python train.py
```

Some available options:
* `--dataset`: Training and testing dataset, support datasets: livec | koniq-10k | bid | live | csiq | tid2013.
* `--train_patch_num`: Sampled image patch number per training image.
* `--test_patch_num`: Sampled image patch number per testing image.
* `--batch_size`: Batch size.

When training on CSIQ dataset, please put 'csiq_label.txt' in your own CSIQ folder.

### Testing on IQA databases

Testing our model on the LIVE Challenge Dataset.

```
python eval.py
```

### curd regression

Curd regression.

```
curd_cpp_matlab/main.cpp

regress.py
```
