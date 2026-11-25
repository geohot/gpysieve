\\ Feed print(checkrow(p,a,b)) for rows of a D=-7 pool to gp -fq check_lift.gp
checkrow(p,a,b) = {
  my(C=Mod(1,p)*[1,-2;1,0],f=factor(b));
  if(!isprime(p)||znorder(Mod(2,p))!=a||(C^b)[2,1]!=0||valuation(a,2)!=1||valuation(b,2)!=2||kronecker(-7,p)!=-1,return(0));
  for(i=1,matsize(f)[1],if((C^(b/f[i,1]))[2,1]==0,return(0)));
  1
};
