#pragma once
#include "stdafx.h"
#include "cytnx.hpp"
#include "utils/vec_unique.hpp"
using namespace cytnx;
//vec_where_all: return all elements'indices same as key
template<class T>
std::vector<int> vec_where_all(const std::vector<T>& in, const T &key){
    std::vector<int> out;
    for(int i = 0; i<in.size(); i++) if (in[i] == key) out.push_back(i);
    return out;
}
std::vector<cytnx_int64> argsort(const std::vector<cytnx_int64> in);
std::tuple<UniTensor, std::vector<int>, std::vector<int>> partial_trace(UniTensor &A, std::vector<int> A_label);
int check_inputs(std::vector<std::vector<int>> connect_list, std::vector<int> flat_connect, std::vector<int> dims_list, std::vector<cytnx_int64> cont_order);
UniTensor ncon(const std::vector<UniTensor> &tensor_list_in, const std::vector<std::vector<int>> &connect_list_in, const bool check_network=false, std::vector<int> cont_order=std::vector<int>(), const std::vector<cytnx_int64>& out_labels=std::vector<cytnx_int64>());
