%module dsa


%{
#include "dsa.h"
#include <thread>
#include <vector>
#include <cmath>
%}

%begin %{
#define SWIG_PYTHON_2_UNICODE
%}

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
  %template(vectorvectori) vector< vector<int> >;
  %template(vectorvectord) vector< vector<double> >;
}

/* http://www.swig.org/Doc1.3/Arguments.html */
%include "typemaps.i"
%{
extern void get_test_train_dist(const d2* ats1, const d2* ats2, const i1* ats1_Ys, const i1* ats2_Ys, i1* min_d_indices);
extern void get_train_train_dist(const d2* ats, const i1* ats_Ys, i1* min_d_indices);
%}
extern void get_test_train_dist(const d2* INPUT, const d2* INPUT, const i1* INPUT, const i1* INPUT, i1* INOUT);
extern void get_train_train_dist(const d2* INPUT, const i1* INPUT, i1* INOUT);

%include "dsa.h"
