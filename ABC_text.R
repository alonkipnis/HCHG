# The ABC test
### Code for P1: Alternatives to the hazard ratio - 
### a simulation-based review

## Part III: A resampling-based test for two crossing survival curves
hatS=function(sample,Time){
  # Return Kaplan-Meiter estimator at t=Time
  
  # input: sample: a observed survival sample (X,delat), where X=min(T,C).
  #         Time£ºThe time points at which that we want to estiamte, 
  #               can be avector or a signel time point. 
  
  surFit=survfit(Surv(sample[,1],sample[,2])~1,error="greenwood")
  surEst=stepfun(surFit$time,c(1,surFit$surv))
  
  # Estimate
  hatSKM=surEst(Time)
  return(hatSKM)
}

hatGamma=function(t,deltaN1,deltaN2,Y1,Y2,index1,index2,n){
  ## Compute hatGamma(t)
  ## input: t is the given time 
  ##        deltaN1: Number of failure at failure time point of sample 1
  ##        deltaN2: Number of failure at failure time point of sample 2
  ##        Y1     : Number of at risk at failure time point of sample 1
  ##        Y2     : Number of at risk at failure time point of sample 2
  ##        index1 : Failure time points of sample 1 less or equal to t
  ##        index2 : Failure time points of sample 2 less or equal to t
  
  ## Gamma1t=sum_{t_i<=t}[DeltaN1(t_i)/{(Y1(t_i)-DeltaN1(t_i))*Y1(t_i)}]
  gamma1t=deltaN1[index1]/((Y1[index1]-deltaN1[index1])*Y1[index1])
  ## Gamma2t=sum_{t_i<=t}[DeltaN2(t_i)/{(Y2(t_i)-DeltaN2(t_i))*Y2(t_i)}]
  gamma2t=deltaN2[index2]/((Y2[index2]-deltaN2[index2])*Y2[index2])
  ## Adjust the infinity value to 0
  gamma1t[gamma1t==Inf]=0
  gamma2t[gamma2t==Inf]=0
  
  hatGamma_t=n*(sum(gamma1t)+sum(gamma2t))
  return(hatGamma_t)}

library(MASS)
scaledABC=function(sample1,sample2,alpha)
{ # return test statistics T_n, upper alpha quantile 
  # of its bootstrap distribution,
  # and upper alpha quantile of its asymptotic distribution
  
  # input: 
  #    sample1: survival data of sample 1
  #    sample2: survival data of sample 2
  #    alpha  : the significant level 
  
  n1=dim(sample1)[1];n2=dim(sample2)[1];n=n1+n2
  
  K=9.4
  
  # Failure time points for sample 1, sample2, and pooled samples
  failureTime1=sort(sample1[sample1[,2]==1,1])
  failureTime2=sort(sample2[sample2[,2]==1,1])
  pooledSample=rbind(sample1, sample2)
  pooledFailureTime=sort(pooledSample[pooledSample[,2]==1,1])
  pooledFailureTime = pooledFailureTime[pooledFailureTime < K]
  numPooledFailure=length(pooledFailureTime)
  
  
  # fit survival curves by KM
  sur.fit1=survfit(Surv(sample1[,1],sample1[,2])~1,error="greenwood")
  sur.fit2=survfit(Surv(sample2[,1],sample2[,2])~1,error="greenwood")
  
  # create KM estimators by step function
  sur.est1=stepfun(sur.fit1$time,c(1,sur.fit1$surv))
  sur.est2=stepfun(sur.fit2$time,c(1,sur.fit2$surv))
  
  # Estimate S_j(t) at failure time points
  S1=sur.est1(pooledFailureTime);
  S2=sur.est2(pooledFailureTime);
  
  # gap time t_{i+1}-t_{i}
  gapTime=diff(c(pooledFailureTime,K))
  
  # T_n=sqrt(n)*sum_{i=1}^{kn}|s1(t_i)-s2(t_i)|*(t_{i+1}-t_i)
  Tn=sqrt(n)*sum(abs(S1-S2)*gapTime)
  
  # quantile of Tn by bootstrapping
  B=2000; bootTn=rep(0,B)
  
  for(iB in 1:B){
    # make the bootstrap procedure replicable
    set.seed(iB)
    # Bootstrap sample 1
    bootSample1=sample1[sample(n1,replace = TRUE),]
    
    set.seed(-iB)
    # Bootstrap sample 2
    bootSample2=sample2[sample(n2,replace = TRUE),]
    
    # Tn*=sqrt(n)*sum_{i=1}^{kn}|{S1*(t_i)-S1(t_i)}
    # -{S2*(t_i)-S2(t_i)}|*(t_{i+1}-t_i)
    bootTn[iB]=sqrt(n)*sum(abs( (hatS(bootSample1,pooledFailureTime)-S1)
                                -(hatS(bootSample2,pooledFailureTime)-S2) )
                           *gapTime)
  } # end of for boot
  quantileBootTn=quantile(bootTn,(1-alpha)) 
  
  return(res=list(Tn=Tn,quantileBootTn=quantileBootTn,
                  bootP=mean(bootTn>=Tn)))
}
