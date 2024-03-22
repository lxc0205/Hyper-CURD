#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include "tools.h"
using namespace std;

void main()
{
    string filename, outname;
    // training datasets
    filename = "./outputs/tid2013.txt";
    // training data matrics
    vector<vector<double>> data(loadMat(filename));
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
    // output dir
    outname = "./outputs/sw_7.txt";
    // conditional uncorrelation process
    IQAProcess_cross(Mssim, mos, 0, 8 * m - 15, outname);
    return;
}