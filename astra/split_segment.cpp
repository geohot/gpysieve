// Segmented exact split-factor sieve up to 2^32-1. Build -O3 -fopenmp.
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
using U=uint64_t;using I=uint32_t;using W=__uint128_t;
struct Mod{U p,rec;explicit Mod(U n):p(n),rec((W(1)<<64)/n){}
 U red(U x)const{U r=x-U(W(x)*rec>>64)*p;return r>=p?r-p:r;}
 U pow(U x,U k)const{U y=1;for(;k;k>>=1){if(k&1)y=red(y*x);x=red(x*x);}return y;}
 U u(U k,int Q)const{U x=0,y=1;for(U bit=U(1)<<(63-__builtin_clzll(k));bit;bit>>=1){U a=red(x*red(x+2*y)),b=red(red(y*y)+p-red(red(p+Q)*red(x*x)));x=a;y=b;if(k&bit){x=red(a+b);y=red(p-red(red(p+Q)*a));}}return x;}
};
static int jac(unsigned a,unsigned n){int s=1;while(a){while(!(a&1)){a>>=1;if(n%8==3||n%8==5)s=-s;}if((a&n&3)==3)s=-s;std::swap(a,n);a%=n;}return n==1?s:0;}
constexpr int ds[]={5,-7,-11,13,-15,17,-19,21};
int main(int argc,char**argv){if(argc!=3)return 1;U lo=strtoull(argv[1],nullptr,10),hi=strtoull(argv[2],nullptr,10);if(lo<21||hi<=lo||hi>UINT32_MAX)return 2;
 std::vector<I> pr;std::vector<bool> composite(65537);for(I i=2;i<=65536;++i)if(!composite[i]){pr.push_back(i);if(i<=256)for(I j=i*i;j<=65536;j+=i)composite[j]=true;}
 U small[8][33]{};int sy[8][22]{};for(unsigned z=0;z<8;++z){auto D=ds[z];for(int j=0;j<abs(D);++j)sy[z][j]=jac(j,abs(D));int64_t x=0,y=1;for(int i=0;i<=32;++i){small[z][i]=std::abs(x);auto t=y-(1-D)/4*x;x=y;y=t;}}
 // Enumerate s with lo < p=2s+1 <= hi.
 U first=(lo+1)/2,end=(hi+1)/2;constexpr U chunk=65536;
 #pragma omp parallel for schedule(dynamic,1)
 for(int block=0; block<int((end-first+chunk-1)/chunk);++block){U base=first+block*chunk,N=std::min(chunk,end-base);std::vector<uint8_t> bad(N);std::vector<int> idx(N,-1);
   for(I q:pr){if(U(q)*q>2*(base+N))break;if(q==2)continue;U off=((q-1)/2+q-base%q)%q;for(U j=off;j<N;j+=q)if(2*(base+j)+1!=q)bad[j]=1;}
   struct Entry{I p,rem;unsigned count;std::array<I,10> fs;};std::vector<Entry> v;v.reserve(N/6);
   for(U j=0;j<N;++j)if(!bad[j]){U s=base+j;idx[j]=v.size();v.push_back({I(2*s+1),I(s>>__builtin_ctzll(s)),1,{2}});}
   for(I q:pr){if(U(q)*q>base+N)break;if(q==2)continue;for(U j=(q-base%q)%q;j<N;j+=q)if(idx[j]>=0){auto &e=v[idx[j]];e.fs[e.count++]=q;do{e.rem/=q;}while(e.rem%q==0);}}
   for(auto &e:v){if(e.rem>1)e.fs[e.count++]=e.rem;U p=e.p,a=p-1;Mod mod(p);for(unsigned j=0;j<e.count;++j){I f=e.fs[j];while(!(a%f)&&mod.pow(2,a/f)==1)a/=f;}
     U bound=std::gcd(p-1,2*((p-1)/a));for(unsigned z=0;z<8;++z){int D=ds[z],Q=(1-D)/4;if(sy[z][p%abs(D)]!=1)continue;
       if(bound<=32?(small[z][bound]%p!=0):bool(mod.u(bound,Q)))continue;
       U b=bound;for(unsigned j=0;j<e.count;++j){I f=e.fs[j];while(!(b%f)&&!mod.u(b/f,Q))b/=f;}
       if(2%std::gcd(a,b))continue;
       #pragma omp critical
       std::printf("%d %u %u %u\n",D,e.p,(I)a,(I)b);
     }
   }
 }
}
