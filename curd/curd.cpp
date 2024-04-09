#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include "utils.h"
using namespace std;

void main()
{  
    string file_name = "tid2013_koniq-10k";
    // training data matrics
    vector<vector<double>> data(loadMat("../outputs/" + file_name + ".txt"));
    // merge sizes
    int m = data[0].size();
    int n = data.size();
    // get Mssim and mos  
    vector<double>mos(n, 0);
    vector<vector<double>> Mssim(m - 1, mos);
    for (int i = 0; i < m - 1; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Mssim[i][j] = data[j][i];
        }
    }
    for (int i = 0; i < n; i++)
    {
        mos[i] = data[i][m - 1];
    }
    data.clear();
    // conditional uncorrelation process with output dir
    IQAProcess(Mssim, mos, 0, 8 * m - 15, "../outputs/sw_7_" + file_name + ".txt");
    return;
}