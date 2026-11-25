// Enumerate divisors, not integers p or subsets. Bounded discovery only;
// pool_lift.py independently certifies every retained prime and its two periods.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <vector>
#include <iostream>
#include <gmpxx.h>
using U=uint64_t;using W=__uint128_t;
struct Axis{mpz_class t;std::vector<std::pair<U,int>> fac;};
Axis minus,plus; U upper,hm,defect,h,lim;bool left;std::vector<std::pair<U,int>>* fs;
U modpow(U k,U p){U x=2,y=1;for(;k;k>>=1){if(k&1)y=W(y)*x%p;x=W(x)*x%p;}return y;}
U smallg(U v,const Axis& z){return std::gcd(v,mpz_fdiv_ui(z.t.get_mpz_t(),v));}
U luc(U k,U p,unsigned Q){U a=0,b=1;for(U s=U(1)<<(63-__builtin_clzll(k));s;s>>=1){U u=(W(a)*(a+2*b))%p,v=(U(W(b)*b%p)+p-U(W(Q)*(W(a)*a%p)%p))%p;a=u;b=v;if(k&s){a=(u+v)%p;b=(p-U(W(Q)*u%p))%p;}}return a;}
void visit(U d,size_t i){U x=h*d,p=left?2*x+1:4*x-1;
 if(p%8==3){bool d5=p%5==2||p%5==3,d7=p%7==3||p%7==5||p%7==6;
 if((d5||d7) && (h==1||smallg(x,left?minus:plus)==d)){
 U a=left?2*d:2*smallg((p-1)/2,minus), b=left?4*smallg((p+1)/4,plus):4*d;
 if((p-1)/a<=defect && (p+1)/b<=defect && modpow(a,p)==1){
 // P=1: square x->a*x+b with x^2=x-Q (Q=-1 Fibonacci uses Q mod p).
 // Fibonacci via conventional doubling below (no signed Q overflow).
 bool f=false;if(d5){U u=0,v=1;for(U s=U(1)<<(63-__builtin_clzll(b));s;s>>=1){U c=W(u)*(2*v+p-u)%p,e=(U(W(u)*u%p)+U(W(v)*v%p))%p;u=c;v=e;if(b&s){u=e;v=(c+e)%p;}}f=!u;}
 bool g=d7&&!luc(b,p,2);if(f||g)std::cout<<p<<' '<<f<<' '<<g<<'\n';}
 }}
 for(size_t j=i;j<fs->size();++j){auto [q,e]=(*fs)[j];U n=d;if(q>lim/n)break;for(int k=0;k<e && n<=lim/q;++k){n*=q;visit(n,j+1);}}
}
int main(int argc,char**argv){if(argc!=4)return 1;upper=strtoull(argv[1],nullptr,10);hm=strtoull(argv[2],nullptr,10);defect=strtoull(argv[3],nullptr,10);if(upper>U(1)<<60||upper<3)return 2;
 for(Axis*z:{&minus,&plus}){z->t=1;int n;if(!(std::cin>>n))return 3;for(int i=0;i<n;++i){U q;int e;std::cin>>q>>e;z->fac.push_back({q,e});for(int j=0;j<e;++j)z->t*=q;}}
 for(bool m:{true,false}){left=m;fs=&(left?minus:plus).fac;for(h=1;h<=hm;h+=2){if(left && h%4!=1)continue;lim=(left?upper-1:upper+1)/((left?2:4)*h);if(lim)visit(1,0);}}
}
