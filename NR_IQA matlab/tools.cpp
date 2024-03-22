#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include "tools.h"
using namespace std;

// 包含二阶交叉项
void IQAProcess_cross(vector<vector<double>>& mssim, vector<double>& mos, double R_min, double R_max, string filename)
{
    // output file stream
    ofstream outfile;
    outfile.open(filename);
    // max saving number and saving container
    int maxNum = 1000000;
    vector<vector<double>> max_sw;
    // Mssim data
    vector<vector<double>> Mssim(mssim);
    // Mssim size
    int m = Mssim.size();     // layer dim
    int n = Mssim[0].size();  // picture dim
    // expand Mssim
    // first order term
    for (int fun = 0; fun < 7; fun++)
    {
        for (int id = 0; id < m; id++)
        {
            vector<double> mssim_temp;
            for (int i = 0; i < n; i++)
            {
                if (fun == 0)
                    mssim_temp.push_back(pow(Mssim[id][i], 2));
                else if (fun == 1)
                    mssim_temp.push_back(sqrt(Mssim[id][i]));
                else if (fun == 2)
                    mssim_temp.push_back(pow(Mssim[id][i], 3));
                else if (fun == 3)
                    mssim_temp.push_back(cbrt(Mssim[id][i]));
                else if (fun == 4)
                    mssim_temp.push_back(log(Mssim[id][i]));
                else if (fun == 5)
                    mssim_temp.push_back(pow(2, Mssim[id][i]));
                else if (fun == 6)
                    mssim_temp.push_back(exp(Mssim[id][i]));
            }
            Mssim.push_back(mssim_temp);
        }
    }
    // second order interactive term
    for (int id_x = 0; id_x < m; id_x++)
    {
        for (int id_y = id_x + 1; id_y < m; id_y++)
        {
            vector<double> mssim_temp;
            for (int i = 0; i < n; i++)
            {
                mssim_temp.push_back(Mssim[id_x][i] * Mssim[id_y][i]);
            }
            Mssim.push_back(mssim_temp);
        }
    }
    // calculate pearson matrix
    m = Mssim.size();
    vector<double> vec_temp(m + 1, 0);
    vector<vector<double>> R(m + 1, vec_temp);
    vec_temp.clear();
    MatPearson(Mssim, mos, R);
    //start conditional uncorrelation process
    int no = 7; //k
    double r0 = 0.9999;// threshold
    double** Rhere = createMatrix(no + 1, no + 1);
    double** Rxy = createMatrix(no + 1, no + 1);
    double temp = 0;
    for (int i = 0; i < no + 1; i++)
    {
        Rhere[i][i] = 1;
    }
    // time clock
    clock_t startTime, endTime;
    startTime = clock();
    for (int i1 = R_min; i1 <= R_max; i1++)
    {
        cout << "k = 7, R num: " << R_min << " / " << i1 << " / " << R_max << endl;
        Rhere[0][no] = R[i1][m];
        for (int i2 = i1 + 1; i2 <= m - 1; i2++)
        {
            temp = R[i1][i2];
            if (temp > r0)
            {
                continue;
            }
            Rhere[0][1] = temp;
            Rhere[1][no] = R[i2][m];
            for (int i3 = i2 + 1; i3 <= m - 1; i3++)
            {
                temp = R[i1][i3];
                if (temp > r0)
                {
                    continue;
                }
                Rhere[0][2] = temp;
                temp = R[i2][i3];
                if (temp > r0)
                {
                    continue;
                }
                Rhere[1][2] = temp;
                Rhere[2][no] = R[i3][m];
                for (int i4 = i3 + 1; i4 <= m - 1; i4++)
                {
                    temp = R[i1][i4];
                    if (temp > r0)
                    {
                        continue;
                    }
                    Rhere[0][3] = temp;
                    temp = R[i2][i4];
                    if (temp > r0)
                    {
                        continue;
                    }
                    Rhere[1][3] = temp;
                    temp = R[i3][i4];
                    if (temp > r0)
                    {
                        continue;
                    }
                    Rhere[2][3] = temp;
                    Rhere[3][no] = R[i4][m];
                    for (int i5 = i4 + 1; i5 <= m - 1; i5++)
                    {
                        temp = R[i1][i5];
                        if (temp > r0)
                        {
                            continue;
                        }
                        Rhere[0][4] = temp;
                        temp = R[i2][i5];
                        if (temp > r0)
                        {
                            continue;
                        }
                        Rhere[1][4] = temp;
                        temp = R[i3][i5];
                        if (temp > r0)
                        {
                            continue;
                        }
                        Rhere[2][4] = temp;
                        temp = R[i4][i5];
                        if (temp > r0)
                        {
                            continue;
                        }
                        Rhere[3][4] = temp;
                        Rhere[4][no] = R[i5][m];
                        for (int i6 = i5 + 1; i6 <= m - 1; i6++)
                        {
                            temp = R[i1][i6];
                            if (temp > r0)
                            {
                                continue;
                            }
                            Rhere[0][5] = temp;
                            temp = R[i2][i6];
                            if (temp > r0)
                            {
                                continue;
                            }
                            Rhere[1][5] = temp;
                            temp = R[i3][i6];
                            if (temp > r0)
                            {
                                continue;
                            }
                            Rhere[2][5] = temp;
                            temp = R[i4][i6];
                            if (temp > r0)
                            {
                                continue;
                            }
                            Rhere[3][5] = temp;
                            temp = R[i5][i6];
                            if (temp > r0)
                            {
                                continue;
                            }
                            Rhere[4][5] = temp;
                            Rhere[5][no] = R[i6][m];
                            for (int i7 = i6 + 1; i7 <= m - 1; i7++)
                            {
                                temp = R[i1][i7];
                                if (temp > r0)
                                {
                                    continue;
                                }
                                Rhere[0][6] = temp;
                                temp = R[i2][i7];
                                if (temp > r0)
                                {
                                    continue;
                                }
                                Rhere[1][6] = temp;
                                temp = R[i3][i7];
                                if (temp > r0)
                                {
                                    continue;
                                }
                                Rhere[2][6] = temp;
                                temp = R[i4][i7];
                                if (temp > r0)
                                {
                                    continue;
                                }
                                Rhere[3][6] = temp;
                                temp = R[i5][i7];
                                if (temp > r0)
                                {
                                    continue;
                                }
                                Rhere[4][6] = temp;
                                temp = R[i6][i7];
                                if (temp > r0)
                                {
                                    continue;
                                }
                                Rhere[5][6] = temp;
                                Rhere[6][no] = R[i7][m];

                                for (int i = 0; i < no + 1; i++)
                                {
                                    for (int j = 0; j < no + 1; j++)
                                    {
                                        Rxy[i][j] = Rhere[i][j];
                                    }
                                }
                                double recidiag;
                                for (int i = 0; i <= no - 1; i++)
                                {
                                    recidiag = 1 / Rxy[i][i];
                                    for (int j = i + 1; j <= no; j++)
                                    {
                                        temp = Rxy[i][j] * recidiag;
                                        for (int p = j; p <= no; p++)
                                        {
                                            Rxy[j][p] = Rxy[j][p] - Rxy[i][p] * temp;
                                        }
                                    }
                                }
                                double sw = Rxy[no][no]; // omega^2
                                if ((sw <= 1) && (sw > 0))
                                {
                                    if (max_sw.size() < maxNum)
                                    {
                                        vector<double> temp{ double(i1),double(i2),double(i3),double(i4),double(i5),double(i6),double(i7),sw };
                                        max_sw.push_back(temp);
                                    }
                                    else
                                    {
                                        double value = 1;
                                        double index = -1;
                                        for (int w = 0; w < maxNum; w++)
                                        {
                                            if (max_sw[w][7] < value)
                                            {
                                                value = max_sw[w][7];
                                                index = w;
                                            }
                                        }
                                        if (sw < max_sw[index][7])
                                        {
                                            max_sw[index][0] = double(i1);
                                            max_sw[index][1] = double(i2);
                                            max_sw[index][2] = double(i3);
                                            max_sw[index][3] = double(i4);
                                            max_sw[index][4] = double(i5);
                                            max_sw[index][5] = double(i6);
                                            max_sw[index][6] = double(i7);
                                            max_sw[index][7] = sw;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //saving  max_sw data
    for (int i = 0; i < maxNum; i++)
    {
        for (int j = 0; j < no + 1; j++)
        {
            outfile << max_sw[i][j] << ' ';
        }
        outfile << endl;
    }
    // close file stream
    outfile.close();
    // end time
    endTime = clock();
    cout << "Time used: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    // free space
    freeMatrix(no + 1, Rhere);
    freeMatrix(no + 1, Rxy);
}

// 原始版本，不包含二阶交叉项 可能存在语法错误
//void IQAProcess(vector<vector<double>>& mssim, vector<double>& mos, double R_min, double R_max, string filename)
//{
//    ofstream outfile;
//    outfile.open(filename);
//    clock_t startTime, endTime;
//    int maxNum = 50000;
//    vector<vector<double>> max_sw;
//    //    Mssim
//    vector<vector<double>> Mssim(mssim);
//    int m = Mssim.size();
//    int n = Mssim[0].size();
//    for (int fun = 0; fun < 7; fun++)
//    {
//        for (int id = 0; id < m; id++)
//        {
//            vector<double> mssim_temp;
//            for (int i = 0; i < n; i++)
//            {
//                if (fun == 0)
//                    mssim_temp.push_back(pow(Mssim[id][i], 2));
//                else if (fun == 1)
//                    mssim_temp.push_back(sqrt(Mssim[id][i]));
//                else if (fun == 2)
//                    mssim_temp.push_back(pow(Mssim[id][i], 3));
//                else if (fun == 3)
//                    mssim_temp.push_back(cbrt(Mssim[id][i]));
//                else if (fun == 4)
//                    mssim_temp.push_back(log(Mssim[id][i]));
//                else if (fun == 5)
//                    mssim_temp.push_back(pow(2, Mssim[id][i]));
//                else if (fun == 6)
//                    mssim_temp.push_back(exp(Mssim[id][i]));
//            }
//            Mssim.push_back(mssim_temp);
//        }
//    }
//    //    Mssim  y  Ƥ  ѷ  ؾ   R
//    int size = Mssim.size();
//    vector<double> vec_temp(size + 1, 0);
//    vector<vector<double>> R(size + 1, vec_temp);
//    vec_temp.clear();
//    MatPearson(Mssim, mos, R);
//    //ѭ  
//    int no = 7;
//    double r0 = 0.9999;
//    double** Rhere = createMatrix(no + 1, no + 1);
//    double** Rxy = createMatrix(no + 1, no + 1);
//    n = Mssim.size();
//    double temp = 0;
//    for (int i = 0; i < no + 1; i++)
//    {
//        Rhere[i][i] = 1;
//    }
//    startTime = clock();//  ʱ  ʼ
//    for (int i1 = R_min; i1 <= R_max; i1++)
//    {
//        cout << i1 << " in C++ 7" << endl;
//        Rhere[0][no] = R[i1][n];
//        for (int i2 = i1 + 1; i2 <= n - 1; i2++)
//        {
//            temp = R[i1][i2];
//            if (temp > r0)
//            {
//                continue;
//            }
//            Rhere[0][1] = temp;
//            Rhere[1][no] = R[i2][n];
//            for (int i3 = i2 + 1; i3 <= n - 1; i3++)
//            {
//                temp = R[i1][i3];
//                if (temp > r0)
//                {
//                    continue;
//                }
//                Rhere[0][2] = temp;
//                temp = R[i2][i3];
//                if (temp > r0)
//                {
//                    continue;
//                }
//                Rhere[1][2] = temp;
//                Rhere[2][no] = R[i3][n];
//                for (int i4 = i3 + 1; i4 <= n - 1; i4++)
//                {
//                    temp = R[i1][i4];
//                    if (temp > r0)
//                    {
//                        continue;
//                    }
//                    Rhere[0][3] = temp;
//                    temp = R[i2][i4];
//                    if (temp > r0)
//                    {
//                        continue;
//                    }
//                    Rhere[1][3] = temp;
//                    temp = R[i3][i4];
//                    if (temp > r0)
//                    {
//                        continue;
//                    }
//                    Rhere[2][3] = temp;
//                    Rhere[3][no] = R[i4][n];
//                    for (int i5 = i4 + 1; i5 <= n - 1; i5++)
//                    {
//                        temp = R[i1][i5];
//                        if (temp > r0)
//                        {
//                            continue;
//                        }
//                        Rhere[0][4] = temp;
//                        temp = R[i2][i5];
//                        if (temp > r0)
//                        {
//                            continue;
//                        }
//                        Rhere[1][4] = temp;
//                        temp = R[i3][i5];
//                        if (temp > r0)
//                        {
//                            continue;
//                        }
//                        Rhere[2][4] = temp;
//                        temp = R[i4][i5];
//                        if (temp > r0)
//                        {
//                            continue;
//                        }
//                        Rhere[3][4] = temp;
//                        Rhere[4][no] = R[i5][n];
//                        for (int i6 = i5 + 1; i6 <= n - 1; i6++)
//                        {
//                            temp = R[i1][i6];
//                            if (temp > r0)
//                            {
//                                continue;
//                            }
//                            Rhere[0][5] = temp;
//                            temp = R[i2][i6];
//                            if (temp > r0)
//                            {
//                                continue;
//                            }
//                            Rhere[1][5] = temp;
//                            temp = R[i3][i6];
//                            if (temp > r0)
//                            {
//                                continue;
//                            }
//                            Rhere[2][5] = temp;
//                            temp = R[i4][i6];
//                            if (temp > r0)
//                            {
//                                continue;
//                            }
//                            Rhere[3][5] = temp;
//                            temp = R[i5][i6];
//                            if (temp > r0)
//                            {
//                                continue;
//                            }
//                            Rhere[4][5] = temp;
//                            Rhere[5][no] = R[i6][n];
//                            for (int i7 = i6 + 1; i7 <= n - 1; i7++)
//                            {
//                                temp = R[i1][i7];
//                                if (temp > r0)
//                                {
//                                    continue;
//                                }
//                                Rhere[0][6] = temp;
//                                temp = R[i2][i7];
//                                if (temp > r0)
//                                {
//                                    continue;
//                                }
//                                Rhere[1][6] = temp;
//                                temp = R[i3][i7];
//                                if (temp > r0)
//                                {
//                                    continue;
//                                }
//                                Rhere[2][6] = temp;
//                                temp = R[i4][i7];
//                                if (temp > r0)
//                                {
//                                    continue;
//                                }
//                                Rhere[3][6] = temp;
//                                temp = R[i5][i7];
//                                if (temp > r0)
//                                {
//                                    continue;
//                                }
//                                Rhere[4][6] = temp;
//                                temp = R[i6][i7];
//                                if (temp > r0)
//                                {
//                                    continue;
//                                }
//                                Rhere[5][6] = temp;
//                                Rhere[6][no] = R[i7][n];
//
//                                for (int i = 0; i < no + 1; i++)
//                                {
//                                    for (int j = 0; j < no + 1; j++)
//                                    {
//                                        Rxy[i][j] = Rhere[i][j];
//                                    }
//                                }
//                                double recidiag;
//                                for (int i = 0; i <= no - 1; i++)
//                                {
//                                    recidiag = 1 / Rxy[i][i];
//                                    for (int j = i + 1; j <= no; j++)
//                                    {
//                                        temp = Rxy[i][j] * recidiag;
//                                        for (int p = j; p <= no; p++)
//                                        {
//                                            Rxy[j][p] = Rxy[j][p] - Rxy[i][p] * temp;
//                                        }
//                                    }
//                                }
//                                double sw = Rxy[no][no]; // ޷  ŷ    ϵ    ƽ   omega^2
//                                if ((sw <= 1) && (sw > 0))
//                                {
//                                    if (max_sw.size() < maxNum)
//                                    {
//                                        vector<double> temp{ double(i1),double(i2),double(i3),double(i4),double(i5),double(i6),double(i7),sw };
//                                        max_sw.push_back(temp);
//                                    }
//                                    else
//                                    {
//                                        double value = 1;
//                                        double index = -1;
//                                        for (int w = 0; w < maxNum; w++)
//                                        {
//                                            if (max_sw[w][7] < value)
//                                            {
//                                                value = max_sw[w][7];
//                                                index = w;
//                                            }
//                                        }
//                                        if (sw < max_sw[index][7])
//                                        {
//                                            max_sw[index][0] = double(i1);
//                                            max_sw[index][1] = double(i2);
//                                            max_sw[index][2] = double(i3);
//                                            max_sw[index][3] = double(i4);
//                                            max_sw[index][4] = double(i5);
//                                            max_sw[index][5] = double(i6);
//                                            max_sw[index][6] = double(i7);
//                                            max_sw[index][7] = sw;
//                                        }
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    for (int i = 0; i < maxNum; i++)
//    {
//        for (int j = 0; j < no + 1; j++)
//        {
//            outfile << max_sw[i][j] << ' ';
//        }
//        outfile << endl;
//    }
//    outfile.close();
//    endTime = clock();//  ʱ    
//    cout << "    ʱ  Ϊ: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//    freeMatrix(no + 1, Rhere);
//    freeMatrix(no + 1, Rxy);
//}

void loadData(string filename, vector<double>& data)
{
    ifstream file(filename, ios::in);

    if (!file)
    {
        cout << "load error!" << endl;
    }
    string str;

    int cnt = 0;
    while (!file.eof())
    {
        getline(file, str);
        if (!str.empty())
        {
            data[cnt] = stod(str);
            cnt++;
        }
    }
}

vector<vector<double>> loadMat(string filename)
{
    ifstream file(filename, ios::in);
    if (!file)
    {
        cout << "load error!" << endl;
    }
    string str;
    vector<vector<double>> mat;
    int cnt = 0;
    while (!file.eof())
    {
        getline(file, str);
        if (!str.empty())
        {
            int pos = 0;
            vector<double> temp;
            for (int i = 0; i < str.length(); i++)
            {
                if (str[i]=='\t')
                {
                    string str0 = str.substr(pos,i - pos);
                    pos = i + 1;
                    temp.push_back(stod(str0));
                }
            }
            mat.push_back(temp);
        }
    }
    return mat;
}

double Pearson(vector<double>& X, vector<double>& Y)
{
    vector<double>X0(X);
    vector<double>Y0(Y);
    int len = X0.size();

    double sum_X = 0, sum_Y = 0,sum_XY = 0, square_sum_X = 0, square_sum_Y = 0;  
    for (int i = 0; i < len; i++)
    {
        sum_X += X0[i];
        sum_Y += Y0[i];
        sum_XY += X0[i] * Y0[i];
        square_sum_X += X0[i] * X0[i];
        square_sum_Y += Y0[i] * Y0[i];
    }
    if (sqrt((len * square_sum_X - sum_X * sum_X) * (len * square_sum_Y - sum_Y * sum_Y)) == 0)
    {
        if (len * sum_XY - sum_X * sum_Y)
        {
            return 1;
        }
    }
    return (len * sum_XY - sum_X * sum_Y)/ sqrt((len * square_sum_X - sum_X * sum_X)* (len * square_sum_Y - sum_Y * sum_Y));
}

void MatPearson(vector<vector<double>> Mat, vector <double> vec, vector<vector<double>>& R_xy)
{
    int size = Mat.size();
    for (int i = 0; i < size; i++)
    {
        for (int j = i; j < size; j++)
        {
            double corr = Pearson(Mat[i], Mat[j]);
            R_xy[i][j] = corr;
            if (i != j)
            {
                R_xy[j][i] = corr;
            }
        }
    }
    for (int i = 0; i < size; i++)
    {
        double corr = Pearson(Mat[i], vec);
        R_xy[i][size] = corr;
        R_xy[size][i] = corr;
    }
    R_xy[size][size] = Pearson(vec, vec);
}

double** createMatrix(int row,int col)
{
    double** TwoDementionalArray = new double* [row];
    for (int i = 0; i < row; i++)
    {
        TwoDementionalArray[i] = new double[col];
    }
    return TwoDementionalArray;
}

void freeMatrix(int row, double** Matrix)
{
    for (int i = 0; i < row; i++)
    {
        delete [] Matrix[i];
        Matrix[i] = NULL;
    }
    delete[] Matrix;
    Matrix = NULL;
}

double* createVector(int size)
{
    double* SingleDementionalArray = new double[size];
    return SingleDementionalArray;
}