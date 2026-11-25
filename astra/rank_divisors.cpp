// Input p Q t num_primes primes... . Output certified lower/upper rank bounds.
// Complete primality/coverage of supplied ell list is handled by caller. t is
// accepted only after checking that C^t is scalar modulo p (gcd(Q,p)=1).
#include <gmpxx.h>
#include <iostream>
#include <vector>
using Z=mpz_class;
bool scalar(const Z &k,long Q,const Z &p){Z a=0,b=1,x,y;
 if(!Q){a=2;mpz_powm(a.get_mpz_t(),a.get_mpz_t(),k.get_mpz_t(),p.get_mpz_t());return a==1;}
 for(long i=mpz_sizeinbase(k.get_mpz_t(),2)-1;i>=0;--i){x=(a*(a+2*b))%p;y=(b*b-Q*a*a)%p;
 if(mpz_tstbit(k.get_mpz_t(),i)){a=(x+y)%p;b=(-Q*x)%p;}else{a.swap(x);b.swap(y);}}
 return a==0;}
int main(){Z p,t;long Q;int n;while(std::cin>>p>>Q>>t>>n){std::vector<Z> ell(n);for(auto &f:ell)std::cin>>f;
 if(!scalar(t,Q,p)){std::cout<<"0 0\n";continue;}Z lo=1;
 for(auto &f:ell){while(t%f==0 && scalar(t/f,Q,p))t/=f;Z z=t;while(z%f==0){lo*=f;z/=f;}}
 std::cout<<lo<<' '<<t<<'\n';}return 0;}
