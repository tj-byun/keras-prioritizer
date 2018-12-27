#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <chrono>

typedef std::vector<double> d1;
typedef std::vector<int> i1;
typedef std::vector<std::vector<double> > d2;
typedef std::vector<std::vector<int> > i2;
typedef unsigned long ulong;

void get_train_train_dist(const d2* ats, const i1* ats_Ys, i1* min_d_indices);
void get_test_train_dist(const d2* ats1, const d2* ats2, const i1* ats1_Ys, const i1* ats2_Ys, i1* min_d_indices);
