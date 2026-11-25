\\ For inert prime r completing cofactor w, ord_r(2) and rank_r(U) both divide w-1.
completion(w,D,lower) = {
  my(k=w-1,u,g,f,rs=List());
  u=if(D==5,fibonacci(k),polcoef(lift(Mod(x,x^2-x+2)^k),1));
  g=gcd(2^k-1,u);
  if(g>=lower,
    f=factor(g);
    for(i=1,matsize(f)[1],if(f[i,1]>=lower,
      if(!isprime(f[i,1]),error("unproven factor"));listput(rs,f[i,1]))));
  print([g,Vec(rs)])
};
