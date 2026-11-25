\\ The other (inert) prime q must have both ord_q(2) and rank_q(U) dividing p-1.
semicomplete(D,p) = {
  my(k=p-1,Q=(1-D)/4,u,g,minq=(2^64+p-1)\p,rs=List(),f);
  u=if(D==5,fibonacci(k),polcoef(lift(Mod(x,x^2-x+Q)^k),1));
  g=gcd(2^k-1,u);
  if(g>=minq,f=factor(g);for(i=1,matsize(f)[1],if(f[i,1]>=minq,
     if(!isprime(f[i,1]),error("unproven prime divisor"));listput(rs,f[i,1]))));
  print([logint(g,2)+1,Vec(rs)])
};
