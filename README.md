# Hyper-CURD

## Dependencies

```
pip install -r requirements.txt
```

## Usage

### Training the baseline model (Original HyperIQA: training)

```
python train.py --pretrained_dataset <pretrained>
```

Some available options:
* `--pretrained_dataset`: Training dataset, support datasets: koniq-10k | live | csiq | tid2013.

Outputs:
* `pkl`: .\outputs\pretrained\\\<pretrained>.pkl

### Testing the baseline model (Original HyperIQA: testing)

```
python eval.py --dataset <dataset> --pretrained_dataset <pretrained> --curd False
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--pretrained_dataset`: Select the pretrained model.
* `--curd`: The flag of using curd.

Outputs:
* print `SRCC` and `PLCC`

### Testing the baseline model and geting the layer scores

```
python eval.py --dataset <dataset> --pretrained_dataset <pretrained> --curd True
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--pretrained_dataset`: Select the pretrained model.
* `--curd`: The flag of using curd.

Outputs:
* `layer scores`: .\outputs\eval outputs\\\<dataset>.txt or .\outputs\eval outputs\\\<dataset>_\<pretrained>.txt

### Curd layer selecting

```
python curd.py --file_name <file name>
```

Some available options:
* `--file_name`: The name of the layer scores in eval output directory.

Outputs:
* `sw flie`: .\outputs\curd outputs\sw_\<file name>.txt

### Sort the curd outputs

```
python sort.py --file_name <file name> --order <True or False> --save_num <save numbers>
```

Some available options:
* `--file_name`: Input file name.
* `--order`: Ascending(True) or descending(False).
* `--save_num`: Save numbers, default number is 50000.

Outputs:
* `sorted sw flie`: .\outputs\sort outputs\sw_\<file name>_sorted.txt

### Regression

```
python regress.py --dataset  <dataset> --pretrained_dataset <pretrained>
```

Some available options:
* `--dataset`: Regression dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--pretrained_dataset`: The pretrained model, for select sw file.

Outputs:
* `regression ouptuts file`: .\outputs\regress outputs\regress\_\<dataset>.txt or .\outputs\regress outputs\regress\_\<dataset>\_\<pretrained_dataset>.txt

### Nonliear function

```
python nonliear.py --beta <'1.61 0.19 -1.10 -7.78 -0.139 23.921 0.038'> --index <'0 1 2 3 10 15 47'> --dataset tid2013
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--beta`: The cofficient of nonliear regression, default 7 numbers.
* `--index`: The selected layer index, default 7 numbers.