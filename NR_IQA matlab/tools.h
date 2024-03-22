#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
using namespace std;


void loadData(string filename, vector<double>& data);

vector<vector<double>> loadMat(string filename);

double Pearson(vector<double>& X, vector<double>& Y);

void MatPearson(vector<vector<double>> Mat, vector <double> vec, vector<vector<double>>& R_xy);

double** createMatrix(int row, int col);

void freeMatrix(int row, double** Matrix);

double* createVector(int size);

void IQAProcess_cross(vector<vector<double>>& mssim, vector<double>& mos, double R_min, double R_max, string filename);

void IQAProcess(vector<vector<double>>& mssim, vector<double>& mos, double R_min, double R_max, string filename);