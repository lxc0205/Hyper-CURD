Clear-Host
python train.py --dataset koniq-10k
python eval.py --dataset koniq-10k --curd False
python eval.py --dataset koniq-10k --curd True
python regress.py --dataset koniq-10k