// Conditional quadratic characters. Each input row is (bitmask,rhs), conditional
// on selection of its prime. Index n is the mandatory anchor. RHS already
// includes its character. Assumes all primes are units at every tested modulus.
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
using U=uint64_t; using W=__uint128_t; constexpr U RHS=U(1)<<63;
struct Basis{std::array<U,60> row{}; int rank=0;
 bool add(U v){while(v&~RHS){int j=std::countr_zero(v);if(!row[j]){row[j]=v;++rank;return true;}v^=row[j];}return !v;}
};
int n; U p,all,required_any=0,leaves=0,charpass=0,exact=0,nodes=0;std::vector<U> primes,mods,targets;std::vector<std::vector<U>> eq;std::vector<int> order;
void check(U mask){
 if(!mask || !(mask & required_any))return;
 for(U t=mask;t;t&=t-1)for(U v:eq[std::countr_zero(t)])if(std::popcount(v&(mask|RHS))&1)return;
 ++charpass;
 U val=p;for(U t=mask;t;t&=t-1){W tmp=W(val)*primes[std::countr_zero(t)];if(tmp>=W(1)<<64){val=0;break;}val=U(tmp);}if(val)return;
 ++exact;
 for(U v=mask;;){int j=v?std::countr_zero(v):n;U m=mods[j];U x=(j==n?1:p)%m;
   for(U t=(j==n?mask:mask^(U(1)<<j));t;t&=t-1)x=W(x)*primes[std::countr_zero(t)]%m;
   if(x!=targets[j])return;
   if(!v)break;
   v&=v-1;
 }
 std::cout<<"hit "<<mask<<'\n';
}
void enumerate(const Basis& b){
 int d=n-b.rank;leaves+=U(1)<<d;
 // Back-substitute an affine origin and each nullspace generator.
 auto lift=[&](U x){for(int j=n-1;j>=0;--j)if(b.row[j] && (std::popcount(x&b.row[j])&1))x^=U(1)<<j;return x&all;};
 U mask=lift(RHS);std::array<U,60> toggle{};int i=0;
 for(int j=0;j<n;++j)if(!b.row[j])toggle[i++]=lift(U(1)<<j);
 check(mask);for(U k=1;k<(U(1)<<d);++k){mask^=toggle[std::countr_zero(k)];check(mask);}
}
void search(const Basis& b,int idx){
 ++nodes;
 if(n-b.rank<=6 || idx==n){enumerate(b);return;}
 int j=order[idx];U bit=U(1)<<j;Basis left=b;
 if(left.add(bit))search(left,idx+1);
 Basis right=b;if(!right.add(bit|RHS))return;
 for(U e:eq[j])if(!right.add(e))return;
 search(right,idx+1);
}
int main(){int fixed;if(!(std::cin>>n>>p>>required_any>>fixed)||n<1||n>58)return 1;all=(U(1)<<n)-1;eq.resize(n+1);Basis b;bool ok=true;
 for(int i=0;i<fixed;++i){U mask;int rhs;std::cin>>mask>>rhs;if(!b.add(mask|(rhs?RHS:0)))ok=false;}
 for(int i=0;i<=n;++i){U q,m,target;int k;if(!(std::cin>>q>>m>>target>>k)||!m)return 2;primes.push_back(q);mods.push_back(m);targets.push_back(target);
   for(int j=0;j<k;++j){U mask;int rhs;std::cin>>mask>>rhs;eq[i].push_back(mask|(rhs?RHS:0));}}
 for(int j=0;j<n;++j)order.push_back(j);
 std::stable_sort(order.begin(),order.end(),[](int a,int b){return eq[a].size()>eq[b].size();});
 for(U e:eq[n])if(!b.add(e)){ok=false;break;}if(ok)search(b,0);
 std::cout<<"done "<<nodes<<' '<<leaves<<' '<<charpass<<' '<<exact<<'\n';
}
