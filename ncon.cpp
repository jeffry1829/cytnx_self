#include "ncon.h"

using namespace cytnx;

//argsort: returns the indices that would sort an array.
std::vector<cytnx_int64> argsort(const std::vector<cytnx_int64> in){
    std::vector<cytnx_int64> v(in.size());
    std::iota(v.begin(),v.end(), 0);
    std::sort(v.begin(),v.end(), [&](cytnx_int64 i,cytnx_int64 j){return in[i]<in[j];});
    return v;
}

// std::vector<std::vector<int>> partial_trace(Unitensor &A, std::vector<int> A_label){
std::tuple<UniTensor, std::vector<int>, std::vector<int>> partial_trace(UniTensor &A, std::vector<int> A_label){
// Partial trace on tensor A over repeated labels in A_label
std::vector<int> label = vec_unique(A_label);
    std::vector<int> outlabel;
auto num_cont = A_label.size() - label.size();
    if (num_cont > 0){
        std::vector<std::vector<int>> dup_list;
        for (int i = 0; i < label.size(); i++){
            auto ele = label[i];
            int sum = 0;
            for (int j = 0; j < A_label[j]; j++)
                if (A_label[j] == ele) sum += A_label[j];
            // std::vector<std::vector<int>> tmp;
            // tmp.push_back(vec_where_all(A_label, ele));
            if (sum>1) dup_list.push_back(vec_where_all(A_label, ele));
        }
        // auto cont_ind = Tensor(dup_list).reshape(2*num_cont, order = 'F');
        std::vector<cytnx_uint64> cont_ind;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < dup_list.size(); j++)
                cont_ind.push_back((cytnx_uint64)dup_list[j][i]);

        // std::vector<int> arange;
        // for(int i = 0; i < A_label.size(); i++) arange.push_back(i);
        std::vector<cytnx_uint64> aran(A_label.size());
        std::iota(aran.begin(),aran.end(), 0);

        std::vector<cytnx_uint64> free_ind = vec_erase(aran, cont_ind);
        auto Ashape = A.shape();
        int cont_dim = 1; for (int i = 0; i < num_cont; i++) cont_dim *= Ashape[cont_ind[i]];
        std::vector<cytnx_int64> free_dim;
        for (int i = 0; i < free_ind.size(); i++) free_dim.push_back(Ashape[free_ind[i]]);

        auto B_label = vec_erase(A_label, cont_ind);
        // auto cont_label = vec_unique(A_label[cont_ind]);
        std::vector<int> tmp;
        for(int i = 0; i < A_label.size(); i++) tmp.push_back(A_label[cont_ind[i]]);
        
        std::vector<int> cont_label = vec_unique(tmp);

        int free_dim_prod = 1;  for (int i = 0; i < free_dim.size(); i++)  free_dim_prod *= free_dim[i];
        auto B = zeros({free_dim_prod}); // B is Tensor

        Tensor Aten = A.get_block();
        Aten = Aten.permute(vec_concatenate(free_ind, cont_ind)).reshape(free_dim_prod, cont_dim, cont_dim);
        for (int ip = 0; ip < cont_dim; ip++){
            B = B + Aten(":", ip, ip);
        }
        // outlabel = B_label;
        return std::make_tuple<UniTensor, std::vector<int>, std::vector<int>> (UniTensor(B.reshape(free_dim), 0), {}, {});
    }else{
        // outlabel = A_label;
        Tensor Aten = A.get_block();
        return std::make_tuple<UniTensor, std::vector<int>, std::vector<int>> (UniTensor(A), {}, {});
    }

}

// Check consistancy of NCON inputs"""
int check_inputs(std::vector<std::vector<int>> connect_list, std::vector<int> flat_connect, std::vector<int> dims_list, std::vector<cytnx_int64> cont_order){

    return 0;
}

UniTensor ncon(const std::vector<UniTensor> &tensor_list_in, const std::vector<std::vector<int>> &connect_list_in, const bool check_network/*false*/, std::vector<int> cont_order/*{}*/, const std::vector<cytnx_int64>& out_labels/*{}*/){
    UniTensor out;
    std::vector<std::vector<int>> connect_list = connect_list_in;
    std::vector<UniTensor> tensor_list = tensor_list_in;
    // for(int i = 0 ; i < connect_list_in.size(); i++)
    //     connect_list.push_back(connect_list_in[i]);
    std::vector<int> flat_connect;

    for(int i = 0; i < connect_list.size(); i++){
        auto sublist = connect_list[i];
        for(int j = 0; j < sublist.size(); j++){
            flat_connect.push_back(sublist[j]);
        }
    }

    if (cont_order.size() == 0){
        std::vector<int> positive;
        for(int i = 0; i < flat_connect.size(); i++){
            if (flat_connect[i] > 0)
                positive.push_back(flat_connect[i]);
        }
        cont_order = vec_unique(positive);
    }

    // check inputs if enabled
    // if (check_network){
    //     std::vector<int> dims_list;
    //     check_inputs(connect_list, flat_connect, dims_list, cont_order);
    // }

	// // do all partial traces
	// for (int ele = 0; ele < tensor_list.size(); ele++){
		// int num_cont = connect_list[ele].size() - vec_unique(connect_list[ele]).size();
		// if (num_cont > 0){
			// tensor_list[ele] = partial_trace(tensor_list[ele], connect_list[ele])
			// cont_order = vec_erase(cont_order, vec_intersect(cont_order, cont_ind, true)[1])
		// }
	// }

    // do all binary contractions
    while (cont_order.size() > 0) {
        // identify tensors to be contracted
        auto cont_ind = cont_order[0];
        std::vector<int> locs;
        for (int ele = 0; ele<connect_list.size(); ele++){
            int sum = 0;
            for (int i = 0; i < connect_list[ele].size(); i++)
                if (connect_list[ele][i] == cont_ind) sum += connect_list[ele][i];
            if (sum>0) locs.push_back(ele);
        }
        // do binary contraction
        // cont_many, A_cont, B_cont = vec_intersect(connect_list[locs[0]], connect_list[locs[1]], true, true);
        std::vector<cytnx_uint64> A_cont;
        std::vector<cytnx_uint64> B_cont;        
        std::vector<int> cont_many;     

        vec_intersect_(cont_many, connect_list[locs[0]], connect_list[locs[1]], A_cont, B_cont);

        std::vector<cytnx_int64> alabel;
        std::vector<cytnx_int64> blabel;
        for (cytnx_int64 i = 0; i < tensor_list[locs[0]].labels().size(); i++) alabel.push_back(i);
        for (cytnx_int64 i = alabel.back()+1; i < alabel.back()+1+tensor_list[locs[1]].labels().size(); i++) blabel.push_back(i);
        for (cytnx_int64 i = 0; i < A_cont.size(); i++){
            alabel[A_cont[i]] = blabel.back()+i+1;
            blabel[B_cont[i]] = blabel.back()+i+1;
        }
        UniTensor Ta = tensor_list[locs[0]].relabels(alabel);
        UniTensor Tb = tensor_list[locs[1]].relabels(blabel);

        // tensor_list.push_back(tensordot(tensor_list[locs[0]], tensor_list[locs[1]], axes=(A_cont, B_cont)));
        // std::cout<<Ta.labels()<<std::endl;
        // std::cout<<Tb.labels()<<std::endl;
		
		//std::cout<<"Ta:"<<'\n'<<Ta.shape()<<' '<<Ta.labels()<<'\n';
		//std::cout<<"Tb:"<<'\n'<<Tb.shape()<<' '<< Tb.labels()<<'\n';
		

        tensor_list.push_back(Contract(Ta, Tb));

        // connect_list.append(np.append(np.delete(connect_list[locs[0]], A_cont), np.delete(connect_list[locs[1]], B_cont)))
        connect_list.push_back(vec_concatenate(vec_erase(connect_list[locs[0]], A_cont), vec_erase(connect_list[locs[1]], B_cont)));
                
        // remove contracted tensors from list and update cont_order
        // cont_order = np.delete(cont_order,np.intersect1d(cont_order,cont_many, assume_unique=True, return_indices=True)[1])
        std::vector<cytnx_uint64> idx0;
        std::vector<cytnx_uint64> idx1;   
        std::vector<int> res;    
        // std::cout<< "cont_order : "<< cont_order<<std::endl;
        // std::cout<< "cont_many : "<< cont_many<<std::endl;
        // std::cout<< "tensor_list : "<< tensor_list<<std::endl;
        // std::cout<< "connect_list : "<< connect_list<<std::endl;
        // std::cout<< "locs : "<< locs[0] << locs[1]<<std::endl;
        tensor_list.erase(tensor_list.begin()+locs[1]);
        tensor_list.erase(tensor_list.begin()+locs[0]);
        connect_list.erase(connect_list.begin()+locs[1]);
        connect_list.erase(connect_list.begin()+locs[0]);
        vec_intersect_(res, cont_order, cont_many, idx0, idx1);
        vec_erase_(cont_order, idx0);
    } 

    //do all outer products
    while (tensor_list.size() > 1) {
        tensor_list[tensor_list.size()-2] = Contract(tensor_list[tensor_list.size()-2], tensor_list[tensor_list.size()-1]);
        connect_list[connect_list.size()-2] = vec_concatenate(connect_list[connect_list.size()-2],connect_list[connect_list.size()-1]);
        tensor_list.erase(tensor_list.begin()+tensor_list.size()-1);
        connect_list.erase(connect_list.begin()+connect_list.size()-1);
    }

    // do final permutation
    if (connect_list[0].size() > 0){
        std::vector<cytnx_int64> mconnect_list;
        for (cytnx_int64 i = 0; i < connect_list[0].size(); i++)
            mconnect_list.push_back(-connect_list[0][i]);
        std::vector<cytnx_int64>  outorder = argsort(mconnect_list);
        out = tensor_list[0].permute(outorder);
    }
    else
        out = tensor_list[0];

    if(out_labels.size())out.set_labels(out_labels);
    return out;
}
