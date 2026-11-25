// Enumerate eligible split factors p in (lo,hi] shared across discriminants.
// If a=ord_p(2), b=rank_p(U), then lcm(a,b)|p-1 and gcd(a,b)|2
// imply b | 2(p-1)/a. One short Lucas evaluation usually excludes p.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
using U=uint64_t;using I=uint32_t;
static U powm(U x,U e,U p){U y=1;while(e){if(e&1)y=y*x%p;x=x*x%p;e>>=1;}return y;}
static U lucas(U k,int Q,U p){U u=0,v=1;for(U h=U(1)<<(63-__builtin_clzll(k));h;h>>=1){U a=u*((u+2*v)%p)%p, b=(v*v%p+p-((p+Q)%p)*(u*u%p)%p)%p;u=a;v=b;if(k&h){u=(a+b)%p;v=(p-((p+Q)%p)*a%p)%p;}}return u;}
int main(int argc,char**argv){if(argc!=3)return 1;int lo=atoi(argv[1]),hi=atoi(argv[2]);if(lo<2||hi<=lo||hi>300000000)return 2;
 std::vector<I> spf(hi+1),pr;for(I i=2;i<=I(hi);++i){if(!spf[i]){spf[i]=i;pr.push_back(i);}for(I q:pr){if(q>spf[i]||U(q)*i>U(hi))break;spf[q*i]=q;}}
 const int ds[]={5,-7,9,-11,13,-15,17,-19,21};
 #pragma omp parallel for schedule(dynamic,512)
 for(int idx=0;idx<(int)pr.size();++idx){U p=pr[idx];if(p<=U(lo))continue;U a=p-1,x=a;std::array<I,32> fs{};int count=0;
   while(x>1){U f=spf[x];fs[count++]=f;while(x%f==0)x/=f;while(a%f==0 && powm(2,a/f,p)==1)a/=f;}
   U t=(p-1)/a, bound=std::gcd(p-1,2*t);
   for(int D:ds){if(abs(D)%p==0)break;if(D==9)continue;int Q=(1-D)/4;if(abs(Q)%p==0)continue;
     if(powm(((D%int(p))+int(p))%p,(p-1)/2,p)!=1 || lucas(bound,Q,p)!=0)continue;
     U b=bound;for(int i=0;i<count;++i){U f=fs[i];while(b%f==0 && !lucas(b/f,Q,p))b/=f;}
     if(2%std::gcd(a,b))continue;
     #pragma omp critical
     std::printf("%d %u %u %u\n",D,(I)p,(I)a,(I)b);
   }
 }
}
