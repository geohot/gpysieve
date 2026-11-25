// Every positive K in an admitted CRT progression, up to inclusive cap.
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
using U=uint64_t;using W=__uint128_t;
U power(U e,U n){U a=2,b=1;while(e){if(e&1)b=W(b)*a%n;e>>=1;if(e)a=W(a)*a%n;}return b;}
int main(int argc,char**argv){if(argc!=2)return 1;U cap=std::stoull(argv[1]),cells=0,checked=0,first=0,all=0,rows=0;std::array<U,3> c,o;U r,m;
while(std::cin>>c[0]>>c[1]>>c[2]>>r>>m>>o[0]>>o[1]>>o[2]){
 if(!m||r>=m||!c[0]||c[0]>=c[1]||c[1]>=c[2]||W(cap)*c[2]>=UINT64_MAX||cap>UINT64_MAX-m)throw std::runtime_error("invalid range");++rows;
 for(U k=r?r:m;k<=cap;k+=m){++cells;U p=c[0]*k-1,q=c[1]*k-1,t=c[2]*k-1;if(p<3)continue;bool comp=false;for(U l:{3,5,7,11,13,17,19,23,29,31}){if((p%l==0&&p!=l)||(q%l==0&&q!=l)||(t%l==0&&t!=l)){comp=true;break;}}if(comp)continue;++checked;
 if(o[0]&&power(o[0],p)!=1)continue;++first;if((o[1]&&power(o[1],q)!=1)||(o[2]&&power(o[2],t)!=1))continue;++all;std::cout<<c[0]<<' '<<c[1]<<' '<<c[2]<<' '<<k<<'\n';
 }}if(!std::cin.eof())return 2;std::cerr<<"{\"limit\":"<<cap<<",\"rows\":"<<rows<<",\"K_visits\":"<<cells<<",\"after_trial\":"<<checked<<",\"first_order_pass\":"<<first<<",\"all_order_pass\":"<<all<<"}\n";}
