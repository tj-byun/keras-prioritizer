#include "dsa.h"


void worker(const d2* ats1, const d2* ats2, const i1* ats1_Ys, const i1*
    ats2_Ys, const ulong i_from, const ulong i_to, int skip_same_class, 
    i1* min_d_indices) {
  /* Get distance between vectors in ats1[i_from:i_to] x ats2.
   * `skip_same_class == 1` indicates that activation trace pairs with the same
   * class prediction should be skipped.
   * `skip_same_class == 0` indicates that activation trace pairs with the
   * different class predictions should be skipped.
   * */
  auto vlen = ats1->at(0).size();   // vector length
  for (ulong i = i_from; i < i_to; i++) {
    double min_dist = 999999.9;
    for (ulong j = 0; j < ats2->size(); j++) {
      if ((*ats1_Ys)[i] == (*ats2_Ys)[j] && skip_same_class) {
        continue;
      } else if ((*ats1_Ys)[i] != (*ats2_Ys)[j] && !skip_same_class) {
        continue;
      }
      // calculate l2 norm of vector v1 - v2
      double sum = 0.0;
      for (ulong k = 0; k < vlen; k++) {
        sum += std::pow(ats1->at(i).at(k) - ats2->at(j).at(k), 2);
      }
      auto dist = std::sqrt(sum);
      if (dist < min_dist) {
        (*min_d_indices)[i] = j;
        min_dist = dist;
      }
    }
  }
}

void get_dist_matrix(const d2* ats1, const d2* ats2, const i1* ats1_Ys,
    const i1* ats2_Ys, int skip_same_class, i1* min_d_indices) {
  /* @param ats1: the 1st vector of activation traces (2d vector)
   * @param ats2: the 2nd vector of activation traces (2d vector)
   * @param ats1_Ys: the 1st prediction vector
   * @param ats2_Ys: the 2nd prediction vector
   * @param skip_same_class: Skip the norm computation between the same classes
   *   if set to 1. Otherwise, skip when the prediction of two inputs belong to
   *   different classes.
   * @param min_d_indices: inidices of the minimum distance elements?
   *   #TODO (Vaibhav): please describe how this works
   * */
  std::vector<std::thread> threads;

  /* Distribute the workload to n_buckets number of threads
   * ex) size: 10,  n_buckets: 4  => bucket_size = 3
   * ex) size: 100, n_buckets: 12 => bucket_size = 9
   * */
  auto n_buckets = std::thread::hardware_concurrency();
  auto bucket_size = std::ceil(ats1->size() / (double) n_buckets);

  for (ulong i = 0; i < n_buckets && i * bucket_size < ats1->size(); i++) {
    ulong i_from = i * bucket_size;
    ulong i_to = (i + 1) * bucket_size;
    i_to = i_to > ats1->size() ? ats1->size() : i_to;
    threads.push_back(std::thread(worker, ats1, ats2, ats1_Ys, ats2_Ys,
          i_from, i_to, skip_same_class, min_d_indices));
  }
  for (auto& t: threads) {
    t.join();
  }
  return;
}



void get_test_train_dist(const d2* ats1, const d2* ats2, const i1* ats1_Ys,
    const i1* ats2_Ys, i1* min_d_indices) {
  /* this method will skip AT pairs that have different class predictions */
  get_dist_matrix(ats1, ats2, ats1_Ys, ats2_Ys, 0, min_d_indices);
}

void get_train_train_dist(const d2* ats, const i1* ats_Ys, i1* min_d_indices) {
  get_dist_matrix(ats, ats, ats_Ys, ats_Ys, 1, min_d_indices);
}

