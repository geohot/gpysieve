\\ Independent verifier for strong base-2 MR + strong Lucas-Selfridge.
\\ Usage: gp -fq verify.gp, then verify([factor1, factor2, ...])
bpswtrace(n, verbose=1) = {
  my(s,d,x,mrpass,D=5,g,Q,C,u,v,qk,lpass);
  if(type(n)!="t_INT" || n<3 || n%2==0, error("expected an odd integer >= 3"));
  s=valuation(n-1,2); d=(n-1)>>s; x=Mod(2,n)^d;
  mrpass=(x==1 || x==-1);
  if(verbose,print("MR: n-1=2^",s," * ",d,"; x[0]=",lift(x)));
  for(j=1,s-1,x=x^2; if(verbose,print("x[",j,"]=",lift(x))); if(x==-1,mrpass=1));
  if(issquare(n),if(verbose,print("square: reject"));return([mrpass,0]));
  while(kronecker(D,n)!=-1,
    g=gcd(D,n); if(g>1 && g<n,if(verbose,print("gcd(D,n)=",g));return([mrpass,0]));
    D=if(D>0,-D-2,2-D));
  Q=(1-D)/4; g=gcd(Q,n); if(g>1,if(verbose,print("gcd(Q,n)=",g));return([mrpass,0]));
  s=valuation(n+1,2); d=(n+1)>>s;
  C=(Mod(1,n)*[1,-Q;1,0])^d; u=C[2,1]; v=trace(C); qk=Mod(Q,n)^d;
  if(verbose,print("Lucas: D=",D," P=1 Q=",Q,"; n+1=2^",s," * ",d,"; U[d]=",lift(u)," V[d]=",lift(v)));
  lpass=(u==0 || v==0);
  for(j=1,s-1,v=v^2-2*qk;qk=qk^2;if(verbose,print("V[2^",j,"*d]=",lift(v)));if(v==0,lpass=1));
  [mrpass,lpass]
};
verify(factors) = {
  my(n=1,r);
  if(type(factors)!="t_VEC" || #factors<2,error("need two or more nontrivial factors"));
  for(i=1,#factors,if(type(factors[i])!="t_INT" || factors[i]<=1,error("invalid factor"));n*=factors[i]);
  print("n=",n," factors=",factors);
  if(n%2==0,print("even composite: reject");return(0));
  r=bpswtrace(n);print("MR=",r[1]," strong Lucas=",r[2]," counterexample=",r[1] && r[2]);
  r[1] && r[2]
};
