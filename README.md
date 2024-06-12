# Hyper-CURD

## Dependencies

```
conda create -n curd python=3.8.18
conda activate curd
pip install -r requirements.txt
```

## Usage

### Training the baseline model (Original HyperIQA: training)

```
python hyperTrain.py --predataset <pretrained>
```

Some available options:
* `--predataset`: Training dataset, support datasets: koniq-10k | live | csiq | tid2013.

Outputs:
* `pkl`: .\outputs\pretrained\\\<pretrained>.pkl


### Testing the baseline model (Original HyperIQA: testing) and geting the layer scores (For CURD)

```
python hyperIQA.py --curd <True or False> --predataset <pretrained> --dataset <dataset>
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--predataset`: Select the pretrained model.
* `--curd`: The flag of using curd, False represents the original HyperIQA.


Outputs:
* `curd=False`: print `SRCC` and `PLCC`

* `curd=True, layer scores`: .\outputs\hyperIQA outputs\\\<dataset>.txt or .\outputs\hyperIQA outputs\\\<dataset>_\<pretrained>.txt

### Curd

```
python curd.py --save_num <save numbers> --predataset <pretrained> --dataset <dataset> 
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--predataset`: Select the pretrained model.
* `--save_num`: Save numbers, default number is 50000.

Outputs:
* `fitting ouptuts file`: .\outputs\curd outputs\fitting\_\<dataset>.txt or .\outputs\curd outputs\fitting\_\<dataset>\_\<predataset>.txt

### Nonliear prediction

```
python prediction.py --predataset <pretrained> --dataset <dataset>
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--predataset`: Select the pretrained model.