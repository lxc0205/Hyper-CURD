import argparse
import numpy as np

def savedata(file, mat):
    for i in range(len(mat)):
        file.write(str(mat[i]))
        file.write('\t')
    file.write('\n')

def loadtxt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        fields = line.split('\t')[:-1]
        float_fields = [float(field) for field in fields]
        data.append(float_fields)
    return np.array(data)

def main(config):
    # 读取数据
    input_file = f"./outputs/curd outputs/sw_{config.file_name}.txt"
    output_file = f"./outputs/sort outputs/sw_{config.file_name}_sorted.txt"
    data = loadtxt(input_file)
    with open(output_file, 'w') as file:
        if config.order:
            sorted_indices = np.argsort(data[:, 7], axis=0, kind='mergesort')
        else:
            sorted_indices = np.argsort(data[:, 7], axis=0, kind='mergesort')[::-1]
        sorted_indices = sorted_indices.reshape(-1, 1)
        sorted_indices = np.tile(sorted_indices, (1, data.shape[1]))
        sorted_matrix = np.take_along_axis(data, sorted_indices, axis=0)

        print(sorted_matrix[0,:])
        
        for i in range(config.save_num):
            savedata(file, sorted_matrix[i,:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', dest='file_name', type=str, default='tid2013_koniq-10k')
    parser.add_argument('--order', dest='order', type=bool, default = True, help='Ascending(True) or descending(False)') 
    parser.add_argument('--save_num', dest='save_num', type=int, default=50000, help='Save numbers.') 
    config = parser.parse_args()
    main(config)