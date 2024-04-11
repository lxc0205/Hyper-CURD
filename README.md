# Hyper-CURD

This is the code of Hyper-CURD. Only for personal use.

## Dependencies

- Python 3.6+
- PyTorch 0.4+
- TorchVision
- scipy

### Training the baseline model on datasets (Original HyperIQA: training)

```
python train.py --pretrained_dataset <dataset name>
```

Some available options:
* `--pretrained_dataset`: Training dataset, support datasets: koniq-10k | live | csiq | tid2013.

Outputs:
* pkl file: .\pretrained\\<pretrained_dataset>.pkl

### Testing the baseline model on Dataset (Original HyperIQA: testing)

```
python eval.py --dataset <dataset name> --pretrained_dataset <dataset name> --curd False
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--pretrained_dataset`: Select the pretrained model.
* `--curd`: The flag of using curd.

Outputs:
* print SRCC and PLCC

### Testing the baseline model on Dataset and geting the layer scores by VGG

```
python eval.py --dataset <dataset name> --pretrained_dataset <dataset name> --curd True
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--pretrained_dataset`: Select the pretrained model.
* `--curd`: The flag of using curd.

Outputs:
* layer score file: \<dataset>.txt or \<dataset>_<pretrained_dataset>.txt

### Curd layer selecting

```
curd/curd.cpp
```

Some available options:
* `file_name`: Layer score file name.

Outputs:
* sw line flie: sw_\<no>\_\<dataset>.txt or sw_\<no>\_\<dataset>_<pretrained_dataset>.txt

### Regression

```
python regress.py --dataset  <dataset name> --pretrained_dataset <dataset name>
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--pretrained_dataset`: The pretrained model, for select sw line file.

Outputs:
* regression result file: outputs\results\_\<dataset>\_no\<no>.txt or outputs\results\_\<dataset>\_\<pretrained_dataset>\_no\<no>.txt

Some available options:
* `--dataset`: CURD training dataset, support datasets:  koniq-10k | live | csiq | tid2013.

```
python nonliear.py --dataset tid2013 --beta '1.61 0.19 -1.10 -7.78 -0.139 23.921 0.038' --index '0 1 2 3 10 15 47'
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--beta`: The cofficient of nonliear regression, default 7 numbers.
* `--index`: The selected layer index, default 7 numbers.