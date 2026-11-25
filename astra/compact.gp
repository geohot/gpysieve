\\ Exact gcd from admissible exponents: only construct the smaller-scale operand.
compact(D,p,A,B) = {
  my(Q=(1-D)/4,g,m,u,f,rs=List(),minq=(2^64+p-1)\p);
  if(A<=B, m=2^A-1;
    if(m==1,g=1,g=gcd(m,lift(((Mod(1,m)*[1,-Q;1,0])^B)[2,1]))),
    u=abs(if(D==5,fibonacci(B),polcoef(lift(Mod(x,x^2-x+Q)^B),1)));
    if(u<=1,g=u,g=gcd(u,lift(Mod(2,u)^A)-1)));
  if(g<1,error("unexpected zero Lucas term"));
  if(g>=minq,f=factor(g);for(i=1,matsize(f)[1],if(f[i,1]>=minq,
      if(!isprime(f[i,1]),error("unproven divisor"));listput(rs,f[i,1]))));
  print([logint(g,2)+1,Vec(rs)])
};
