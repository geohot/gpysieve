\\ Offsets H relative to smallest prime x, q=x+H[i]. Requires shape.gp.
constellation(H,D)={my(k=#H,R,S,T,Q=(1-D)/4,M,G,best=-1,b=0,ix=vector(k,i,i),e=vector(k),f,ct=0,vs=List(),x,n);
  for(i=1,k,e[i]=prod(j=1,k,if(j==i,1,1+H[j]-H[i]))-1);ix=vecsort(ix,i->abs(e[i]));
  for(t=1,k,my(i=ix[t]);R=e[i];if(!R,next);S=prod(j=1,k,if(j==i,1,H[j]-H[i]-1))-1;T=R+2;
    if(max(abs(R),max(abs(S),abs(T)))>4000000,next);M=2^abs(R)-1;G=lcm(gcd(M,shapeU(abs(S),Q)),gcd(M,shapeU(abs(T),Q)));if(best<0||G<best,best=G;b=i);if(logint(best,2)<92,break));
  if(best<1||logint(best,2)>150,return([-1,0,0,[]]));f=factor(best);if(factorback(f)!=best,error("factorback"));
  for(j=1,matsize(f)[1],if(!isprime(f[j,1]),error("uncertified anchor"));x=f[j,1]-H[b];if(x<3||!prod(i=1,k,isprime(x+H[i])),next);ct++;n=prod(i=1,k,x+H[i]);if(Mod(2,n)^(n-1)==1,listput(vs,x)));return([b,logint(best,2)+1,ct,Vec(vs)])};
