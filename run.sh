Clear-Host
python train.py --dataset koniq-10k
python eval.py --dataset tid2013 --pretrained_dataset koniq-10k --curd False
python regress.py --dataset koniq-10k