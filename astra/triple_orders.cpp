// Complete base-2-compatible triples, no LCM restriction. Each modular test
// is necessary and sufficient at its selected entry after fixing the 2-part.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <gmpxx.h>
using U=uint64_t;
int main(){std::vector<unsigned>a;std::vector<mpz_class>p;unsigned j;mpz_class q;while(std::cin>>j>>q){if(!j||j>65535||(!a.empty()&&j<a.back()))throw std::runtime_error("order");a.push_back(j);p.push_back(q);}if(a.empty())return 1;
 std::vector<std::vector<uint16_t>>res(a.back()+1);for(auto m:a)if(res[m].empty()){for(auto &x:p)res[m].push_back(mpz_fdiv_ui(x.get_mpz_t(),m));}
 U anchors=0,probes=0,first=0,second=0,all=0;for(size_t i=0;i<a.size();){auto m=a[i],s=m&-m;std::vector<int>inv(m,-1);for(unsigned x=1;x<m;++x){mpz_class z;mpz_class X=x,M=m;if(mpz_invert(z.get_mpz_t(),X.get_mpz_t(),M.get_mpz_t()))inv[x]=z.get_ui();}
 std::vector<std::vector<size_t>> bins(m);std::vector<size_t>js;auto &R=res[m];for(size_t k=0;k<i;k++)if((a[k]&-a[k])==s && inv[R[k]]>=0){bins[R[k]].push_back(k);js.push_back(k);}do{
 ++anchors;for(auto b:js){++probes;const auto mb=a[b];for(auto c:bins[inv[R[b]]]){if(c>=b)break;++first;if(U(res[mb][i])*res[mb][c]%mb!=1)continue;++second;const auto mc=a[c];if(U(res[mc][i])*res[mc][b]%mc!=1)continue;++all;std::cout<<c<<' '<<b<<' '<<i<<'\n';}}bins[1].push_back(i);js.push_back(i);++i;}while(i<a.size()&&a[i]==m);}
 std::cerr<<"{\"entries\":"<<p.size()<<",\"anchors\":"<<anchors<<",\"pair_probes\":"<<probes<<",\"one_order\":"<<first<<",\"two_orders\":"<<second<<",\"all_three\":"<<all<<"}\n";}
