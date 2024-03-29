//#pragma GCC optimize(1)
//#pragma GCC optimize(2)
#pragma GCC optimize(3)
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
#pragma comment(linker, "/STACK:1024000000,1024000000")
#include <bits/stdc++.h>
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
const int _n=15,_d=5,_num=1e3;
int t,n,tot_bond_cnt=0;
VI flatlink,flatbonds;
vector<VI> bonds;
int main(void) {//ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
	//ifstream cin("input.txt");
    ofstream cout("output.txt");
	n=rnd(_n)+1;
	cout<<n<<'\n';
	rep(_,0,n){
		int d=rnd(_d)+1;
		int dd=0;
		VI l;
		while(dd!=d){
			int li=rnd(d-dd)+1;
			dd+=li;
			l.pb(li);
			flatbonds.pb(li);
		}
		bonds.pb(l);
		tot_bond_cnt+=SZ(l);
	}
	int cont_cnt=rnd((int)tot_bond_cnt/((int)2)),neg_bond_cnt=tot_bond_cnt-2*cont_cnt;
	rep(i,1,cont_cnt+1)rep(__,0,2)flatlink.pb(i);
	rep(i,1,neg_bond_cnt+1)flatlink.pb(-i);
	shuffle(all(flatlink),mrand);

	VI last_cont_idx(cont_cnt+1,-1);
	per(i,0,tot_bond_cnt)if(flatlink[i]>0 and last_cont_idx[flatlink[i]]==-1)last_cont_idx[flatlink[i]]=i;
	
	int idx0=0;
	rep(i,0,SZ(bonds)){
		int ele_cnt=1;
		rep(j,0,SZ(bonds[i])){
			if(flatlink[idx0+j]>0 and last_cont_idx[flatlink[idx0+j]]!=idx0+j)flatbonds[idx0+j]=flatbonds[last_cont_idx[flatlink[idx0+j]]];
			bonds[i][j]=flatbonds[idx0+j];
			ele_cnt*=bonds[i][j];
		}

		idx0+=SZ(bonds[i]);

		cout<<SZ(bonds[i])<<'\n';
		rep(j,0,SZ(bonds[i]))cout<<bonds[i][j]<<' ';
		cout<<'\n';
		rep(j,0,ele_cnt)cout<<rnd(_num)<<' ';
		cout<<'\n';
	}	

	vector<VI> links;
	int idx=0;
	rep(i,0,SZ(bonds)){
		VI link;
		rep(j,0,SZ(bonds[i]))link.pb(flatlink[idx+j]);
		idx+=SZ(bonds[i]);
		links.pb(link);
	}

	// ENSURE NO PARTIAL TRACE PRESENTS
	VI pos(cont_cnt+1,-1);
	rep(i,0,SZ(links)){
		rep(j,0,SZ(links[i])){
			if(links[i][j]>0 and pos[links[i][j]]!=-1){
				links[i][pos[links[i][j]]]*=-1;
				links[i][j]*=-1;
			}else if(links[i][j]>0 and pos[links[i][j]]==-1)pos[links[i][j]]=j;
		}
		fill(all(pos),-1);
	}


	rep(i,0,SZ(links)){
		rep(j,0,SZ(links[i]))cout<<links[i][j]<<' ';
		cout<<'\n';
	}
	cout<<0<<'\n'; // cont_order count
	return 0;
}
