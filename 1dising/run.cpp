#include "cytnx.hpp"
#include <iostream>

using namespace std;
using namespace cytnx;

UniTensor GTNO(vector<cytnx_double> params){
    cytnx_double norm_factor = 0;
    for(int i=0;i<params.size();i++){
        norm_factor += params[i]*params[i];
    }
    vector<cytnx_double> cs;
    for(int i=0;i<params.size();i++){
        cs.push_back(params[i]/sqrt(norm_factor));
    }
    Bond bd_in(2,BD_IN), bd_aux(2,BD_OUT);
    UniTensor G = UniTensor({bd_in, bd_in.redirect(), bd_aux.redirect(), bd_aux}, {"p","q","L","R"}, -1, Type.ComplexDouble);
    // G = torch.zeros((2,2,2,2,2,2),dtype=cfg.global_args.torch_dtype,device='cpu')
    UniTensor Id = UniTensor({bd_in, bd_in.redirect()}, vector<string>({"p","q"}), -1, Type.ComplexDouble);
    Id.get_block_().at({0,0}) = 1;
    Id.get_block_().at({1,1}) = 1;
    // UniTensor X = UniTensor({bd_in, bd_in.redirect()}, vector<string>({"p","q"}), -1, Type.ComplexDouble);
    // X.get_block_()(0,1) = 1;
    // X.get_block_()(1,0) = 1;
    UniTensor Z = UniTensor({bd_in, bd_in.redirect()}, vector<string>({"p","q"}), -1, Type.ComplexDouble);
    Z.get_block_().at({0,0}) = 1;
    Z.get_block_().at({1,1}) = -1;

    G.get_block_()(":",":",0,0) = params[0]*Id.get_block_();
    G.get_block_()(":",":",0,1) = params[1]*Z.get_block_();
    G.get_block_()(":",":",1,0) = params[1]*Z.get_block_();
    G.get_block_()(":",":",1,1) = params[2]*Id.get_block_();

    return G;
}

int main() {
    int L = 10, phy = 2, chi = 16;
    Bond bd_in(phy,BD_IN), bd_aux(1,BD_OUT);
    UniTensor A = UniTensor({bd_aux.redirect(), bd_in, bd_aux}, {"L","p","R"}, Type.ComplexDouble);
    random::Make_uniform(A);
    
    
}