// Exact subset join for a mandatory inert prime and <=48 smaller core primes.
// stdin: p a b followed by q c d inverse(q mod lcm(a,b)) rows (units only).
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
using U=uint64_t; using I=uint32_t; using W=__uint128_t;
struct Row { U q,c,d,inv; };
static U mix(U x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
int main(){
 U p,a,b; if(!(std::cin>>p>>a>>b))return 1; std::vector<Row> rows; Row r;
 while(std::cin>>r.q>>r.c>>r.d>>r.inv)rows.push_back(r);
 if(rows.size()>48||!a||!b||W(a)*b>=W(1)<<62)return 2;
 U m=std::lcm(a,b); unsigned k=rows.size()/2; I L=1u<<k;
 std::vector<U> keys(2*L); std::vector<I> head(2*L),link(L);
 auto slot=[&](U key){I h=mix(key)&(2*L-1);while(keys[h]&&keys[h]!=key)h=(h+1)&(2*L-1);return h;};
 U pass=0,two=0,full=0;
 auto visit=[&](U mask){
   if(!mask)return;
   U value=p;
   for(U v=mask;v;v&=v-1){W n=W(value)*rows[std::countr_zero(v)].q;if(n>=W(1)<<64){value=0;break;}value=U(n);}
   if(value)return;
   ++pass; bool first=true;
   for(U v=mask;v;){unsigned j=std::bit_width(v)-1;v^=U(1)<<j;
     U mod=std::lcm(rows[j].c,rows[j].d), rem=p%mod;
     for(U t=mask^(U(1)<<j);t;t&=t-1)rem=W(rem)*rows[std::countr_zero(t)].q%mod;
     if(rem!=1%mod)return;
     if(first){++two;first=false;}
   }
   ++full;std::cout<<"hit "<<mask<<'\n'<<std::flush;
 };
 auto left=[&](auto&& f,unsigned i,U val,I mask,unsigned parity)->void{
   if(i==k){U key=2*val+parity+1;I h=slot(key);keys[h]=key;link[mask]=head[h];head[h]=mask+1;return;}
   f(f,i+1,val,mask,parity);f(f,i+1,W(val)*rows[i].q%m,mask|(1u<<i),1-parity);
 };
 auto right=[&](auto&& f,unsigned i,U val,U mask,unsigned parity)->void{
   if(i==rows.size()){U key=2*val+parity+1;for(I j=head[slot(key)];j;j=link[j-1])visit(mask|(j-1));return;}
   f(f,i+1,val,mask,parity);f(f,i+1,W(val)*rows[i].inv%m,mask|(U(1)<<i),1-parity);
 };
 left(left,0,1%m,0,0);right(right,k,1%m,0,0);
 std::cout<<"done "<<pass<<' '<<two<<' '<<full<<'\n';
}
