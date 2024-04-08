Clear-Host
python train_test_IQA.py --dataset koniq-10k
python main.py --dataset koniq-10k --curd False
python main.py --dataset koniq-10k --curd True
python regression.py --dataset koniq-10k