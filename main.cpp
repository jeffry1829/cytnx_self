//#pragma GCC optimize(1)
//#pragma GCC optimize(2)
/*#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
#pragma comment(linker, "/STACK:1024000000,1024000000")*/
#include "stdafx.h"
#include "main.h"
using namespace cytnx;
using namespace std;
//#define int long long
#define rep(i,a,n) for(int i=a;i<n;i++)
#define per(i,a,n) for(int i=n-1;i>=a;i--)
#define pb push_back
//#define mp make_pair
#define all(x) (x).begin(),(x).end()
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define abs(x) (((x)<0)?(-(x)):(x))
typedef vector<int> VI;
typedef long long ll;
typedef pair<int,int> PII;
typedef double db;
mt19937 mrand(random_device{}());
const ll mod=1000000007;
int rnd(int x){return mrand()%x;}
ll powmod(ll a,ll b){ll res=1;a%=mod;assert(b>=0);for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a, ll b){return b?gcd(b,a%b):a;}
inline int pmod(int x, int d){int m = x%d;return m+((m>>31)&d);}
#define y1 ojsapogjahg
#define prev ojaposjdas
#define rank oiajgpowsdjg
#define left aijhgpiaejhgp
//#define end aononcncnccc
//head
const int _n=1e5+10;
int t,n,m,b;
vector<vector<cytnx_int64>> bonds;
vector<VI> links;
vector<UniTensor> uTs;
int main(void) {ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
	ifstream cin("output.txt");

	cin>>n;
	rep(i,0,n){
		vector<cytnx_int64> bond;
		cin>>b;
		rep(j,0,b){
			cin>>t;
			bond.pb(t);
		}
		vector<cytnx_uint64> ubond;
		transform(all(bond),back_inserter(ubond),[](cytnx_int64 x){return (cytnx_uint64)x;});
		bonds.pb(bond);
		Tensor T=zeros(ubond);
		int tot_dim=1;
		rep(j,0,SZ(bond))tot_dim*=bond[j];
		T.reshape_(tot_dim);
		rep(j,0,tot_dim){
			cin>>t;
			T(j)=t;
		}
		T.reshape_(bond);
		UniTensor uT=UniTensor(T,0);
		uTs.pb(uT);
	}
	rep(i,0,n){
		VI link;
		rep(j,0,SZ(bonds[i])){
			cin>>t;
			link.pb(t);
		}
		links.pb(link);
	}
	UniTensor res=ncon(uTs,links);
	int tot_dim=1;

	std::cout<<"IM HERE3"<<std::endl;

	vector<cytnx_uint64> shape=res.shape();
	rep(i,0,SZ(shape))tot_dim*=shape[i];
	vector<cytnx_int64> vtot_dim;
	vtot_dim.pb(tot_dim);
	res.reshape_(vtot_dim);
	rep(i,0,tot_dim)cout<<res.get_block()(i).item()<<' ';
	cout<<'\n';
	return 0;
}
