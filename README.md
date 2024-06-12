# Hyper-CURD

## Dependencies

```
conda create -n curd python=3.8.18
conda activate curd
pip install -r requirements.txt
```

## Usage
### Testing the baseline model (Original HyperIQA: testing) and geting the layer scores (For CURD)

```
python hyperIQA.py <--curd> --predataset <pretrained> --dataset <dataset>
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--predataset`: Select the pretrained model.
* `--curd`: The flag of using curd, switch off represents the original HyperIQA.


Outputs:
* `No usage of --curd`: print `SRCC` and `PLCC`

* `Usage of curd, layer scores`: .\outputs\hyperIQA outputs\\\<dataset>.txt or .\outputs\hyperIQA outputs\\\<dataset>_\<pretrained>.txt

### Curd

```
python curd.py --save_num <save_num> --predataset <pretrained> --dataset <dataset> 
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--predataset`: Select the pretrained model.
* `--save_num`: Save numbers, default number is 50000.

Outputs:
* `fitting ouptuts file`: .\outputs\curd outputs\fitting\_\<dataset>.txt or .\outputs\curd outputs\fitting\_\<dataset>\_\<predataset>.txt

### Nonliear prediction

```
python prediction.py <--mode> --predataset <pretrained> --dataset <dataset>
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--predataset`: Select the pretrained model.
* `--mode`: The flag of using HyperIQA network, switch off represents the usage of the loading scores in files.



## Additions

### Training the baseline model (Original HyperIQA: training)

```
python hyperTrain.py --predataset <pretrained>
```

Some available options:
* `--predataset`: Training dataset, support datasets: koniq-10k | live | csiq | tid2013.

Outputs:
* `pkl`: .\outputs\pretrained\\\<pretrained>.pkl