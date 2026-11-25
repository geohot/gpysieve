\\ Certificate sieve for n=(p-u)p(p+v). Local factor q=p+s.
shapeU(k,Q)={my(a=0,b=1,c,e);if(k==0,return(0));if(Q == -1,return(fibonacci(k)));forstep(i=logint(k,2),0,-1,c=a*(2*b-a);e=b^2-Q*a^2;if(bittest(k,i),a=e;b=e-Q*c,a=c;b=e));a};
shapeg(u,v,s,D)={my(other=select(x->x!=s,[-u,0,v]),i=other[1]-s,j=other[2]-s,R=(1+i)*(1+j)-1,S=(i-1)*(j-1)-1,T=R+2,Q=(1-D)/4,M);if(!R,return(-1));M=2^abs(R)-1;return(lcm(gcd(M,shapeU(abs(S),Q)),gcd(M,shapeU(abs(T),Q))))};
shape(u,v,D)={my(ss=[0,-u,v],s,g,f,qs=List(),q,p,ps,n,ct=0,best=-1,pos=0);
  for(j=1,3,s=ss[j];g=shapeg(u,v,s,D);if(g>0 && (best<0 || g<best),best=g;pos=s);if(best>0 && logint(best,2)<92,break));
  if(best<1 || logint(best,2)>150,return([-1,0,0,[]]));
  f=factor(best);if(factorback(f)!=best,error("factorback"));
  for(i=1,matsize(f)[1],q=f[i,1];if(!isprime(q),error("uncertified anchor"));p=q-pos;ps=[p-u,p,p+v];if(ps[1]<3 || !prod(j=1,3,isprime(ps[j])),next);ct++;n=prod(j=1,3,ps[j]);if(Mod(2,n)^(n-1)!=1,next);listput(qs,p));return([pos,logint(best,2)+1,ct,Vec(qs)])};
