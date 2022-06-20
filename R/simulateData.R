#' inv_logit: Returns the inverse logit, or "Sigmoid" of the input
#'
#' @param x Input
#' @return inv_logit(x), or equivalently, Sigmoid(x)
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/dlglm}
#'
#' @export
inv_logit = function(x){
  return(1/(1+exp(-x)))
}

#' simulateData: Simulates covariates matrix X and response Y
#'
#' @param dataset Dataset name (default: "SIM"). Inference and prediction L1 computed just for SIM
#' @param N Number of observations in data
#' @param D Dimensionality of simulated latent variable
#' @param P Number of features in data
#' @param data_types_x P-length vector of types of X ("real", "pos", "cat", "count")
#' @param family Family of response variable: "Gaussian", "Multinomial", or "Poisson" (not tested)
#' @param seed Seed to generate reproducible data
#' @param NL_x Boolean: nonlinear generation of X from Z
#' @param NL_y Boolean: nonlinear generation of Y from X
#' @param beta Base value of simulated coefficients for generation of Y from X (default: 0.25)
#' @param C Number of categories of "cat" variables in X. All categorical variables in X are simulated with the same number of categories
#' @param Cy Number of categories of categorical variable Y (family = "Multinomial")
#' @return list with data and params
#' @examples
#' EXAMPLE HERE
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/dlglm}
#'
#' @importFrom MASS mvrnorm
#' @importFrom qrnn elu
#'
#' @export
simulateData = function(dataset = "SIM", N, D, P, data_types_x, family, seed, NL_x, NL_y,
                        beta=0.25, C=NULL, Cy=1){
  # if params are specified, then simulate from distributions
  # if file.name specified, then read in data file --> should have "data"
  # seed determined by sim_index

  logsumexp <- function (x) {
    y = max(x)
    y + log(sum(exp(x - y)))
  }
  softmax <- function (x) {
    exp(x - logsumexp(x))
  }

  if(dataset=="HEPMASS"){ # 7M x 27 --> 300K x 21 feats
    # temp <- tempfile()
    f1 = sprintf("./Results_%s/1000_train.csv.gz",dataset)
    f2 = sprintf("./Results_%s/1000_test.csv.gz",dataset)
    if(!file.exists(f1)){ download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_train.csv.gz", destfile=f1) }
    if(!file.exists(f2)){ download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_test.csv.gz", destfile=f2) }
    data1 <- read.csv(gzfile(f1),header=F,skip=1, nrows=700000)
    data2 <- read.csv(gzfile(f2),header=F,skip=1, nrows=350000)

    data=rbind(data1,data2)

    data = data[data[,1]==1,]    # only class 1 (class 0 is noise)

    classes = data[,1]
    data = data[,-1]; data = data[,-ncol(data)]   # first column: class. last column is bad? pre-processed out

    features_to_remove = apply(data,2,function(x) max(table(x))) > 100    # if more than 100 repeats in feature, remove
    data = data[, !features_to_remove]

    data = (data - colMeans(data)) / apply(data,2,sd)   # normalize
    params=NULL

    X=data; Y=classes
  }else if(dataset=="SPAM"){  # 4601 x 57
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"),header=FALSE)
    colnames(data)=c(paste("word",c(1:48),sep=""),paste("char",c(1:6),sep=""),"capital_run_length_average",
                     "capital_run_length_longest","capital_run_length_total","spam")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]    # take out class (2)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="IRIS"){ # 150 x 4
    data <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"),
                     header=FALSE, col.names=c("sepal.length","sepal.width","petal.length","petal.width","species"))
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]     # take out class (3)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="ADULT"){  # 32561 x 14
    data <- read.csv(url("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data"),header=FALSE)
    colnames(data)=c("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",
                     "race","sex","capital-gain","capital-loss","hours-per-week","native-country",'income_class')
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]  # take out class (2)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="WINE"){ # 178 x 13
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"),header=FALSE)
    colnames(data)=c("class","Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols","Flavanoids",
                     "Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280/OD315_of_diluted_wines","Proline")
    classes=data[,1]
    data=data[,-1]     # take out class (3)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="RED"){ # 1599 x 12
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"),header=TRUE,sep=";")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]     # take out class (quality, 1-10)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="WHITE"){ # 4898 x 12
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"),header=TRUE,sep=";")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]     # take out class (quality, 1-10)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="BREAST"){ # 569 x 30
    # Wisconsin BC (Diagnostic)
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"),
                     header=FALSE)
    colnames(data)=c("ID","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                     "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
                     "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                     "concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
                     "area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
    data=data[,-1]       # remove ID's
    classes=data[,1]
    data=data[,-1]      # take out class (2)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="YEAST"){  # 1484 x 8
    data <- read.table(url("https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"),header=FALSE)
    colnames(data)=c("sequence_name","mcg","gvh","alm","mit","erl","pox","vac","nuc","class")
    data=data[,-1]   # remove sequence names
    classes=data[,ncol(data)]
    data=data[,-ncol(data)] # take out class (10)
    params=NULL
    X=data; Y=classes
  } else if(dataset=="CONCRETE"){ # 1030 x 8
    require(RCurl);require(gdata)
    data <- read.xls("http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)] # take out class: here it's continuous
    params=NULL
    X=data; Y=classes
  } else if(dataset=="BANKNOTE"){ # 1372 x 4
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"),header=FALSE)
    classes=data[,ncol(data)]
    data=data[,-ncol(data)] # take out class (2)
    params=NULL
    X=data; Y=classes
  } else if(grepl("SIM",dataset)){

    X = matrix(nrow=N, ncol=P)
    set.seed(seed)

    P_real=sum(data_types_x=="real"); P_cat=sum(data_types_x=="cat"); P_count=sum(data_types_x=="count"); P_pos=sum(data_types_x=="pos")

    ## Using Linear/Nonlinear like P2
    X = matrix(nrow=N, ncol=P)
    Ws = list(); Bs = list()  # save these weights/biases in generation of X from Z
    if(NL_x){

      # N=100000; P=25; seed=9*sim_index
      set.seed(seed)
      H=64

      ## skew-normal X2
      # Z1 = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))
      # Z2 = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))
      Z = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))

      generate_nonlinear_data = function(Z,H,type=c("real","cat","count","pos"), P_type, C=NULL){
        ### tests:
        # type="real"; P_type=5; C=NULL
        # type="cat"; P_type=5; C=3
        # type="count"; P_type=5; C=NULL
        # type="pos"; P_type=5; C=NULL
        ############
        N = nrow(Z); D = ncol(Z)
        if(type %in% c("real", "count","pos")){P0 = P_type} else if(type =="cat"){ P0 = C*P_type}
        # W1 = matrix(runif(D*floor(P0/2),0,1),nrow=D,ncol=floor(P0/2)) # weights
        # W2 = matrix(runif(D*(P0-floor(P0/2)),0,1),nrow=D,ncol=P0-floor(P0/2)) # weights      # B = matrix(rnorm(N*P,0,0.25),nrow=N,ncol=P_type)


        W1 = if(type=="cat"){
          matrix(3*sample(c(-1,1),size=D*H,replace=T),nrow=D,ncol=H)
        }else{
          matrix(rnorm(D*H,0,0.5),nrow=D,ncol=H)
        }
        W2 = if(type=="cat"){
          matrix(3*sample(c(-1,1),size=H*P0,replace=T),nrow=H,ncol=P0)
        }else{
          matrix(rnorm(H*P0,0,0.5),nrow=H,ncol=P0)
        }

        B = matrix(rnorm(N*P0,0,1),nrow=N,ncol=P0)
        # B1 = matrix(rnorm(N*P0,0,1),nrow=N,ncol=H)
        # B2 = matrix(rnorm(N*P0,0,1),nrow=N,ncol=P0)
        # B0 = 0 #location shift?
        B0 = 10

        # eta = qrnn::elu(Z %*% W1 + B1) %*% W2 + B2
        eta = qrnn::elu(Z %*% W1) %*% W2 + B
        eta = apply(eta,2,function(x){(x-mean(x))/sd(x)})  # pre-normalize to mean 0
        eta = eta + B0  # add location shift

        # eta = Z %*% W
        if(type=="real"){
          X = eta  # if B matrix (noise) is applied to eta
        }else if(type=="cat"){
          X = matrix(nrow=N,ncol=0)
          for(j in 1:P_type){
            eta0 = eta[,(C*(j-1)+1):(C*j)]  # subset relevant C columns of linear transformation from Z
            probs = t(apply(eta0, 1, function(x) softmax(x)))
            classes = apply(eta0, 1, function(x){which.max(rmultinom(1,1,softmax(x)))}) # eta0 --> softmax probs --> multinom --> designate class
            X = cbind(X, classes)
          }
        }else if(type=="count"){
          X = matrix(rpois(N*P_type, eta+100), N, P_type)  # add constant of 100 to lambda --> larger counts
        }else if(type=="pos"){
          X = matrix(rlnorm(N*P_type, eta, 0.1), N, P_type) # log-sd changed to have range of about 0 - 80
        }
        # return(list(X=X,W=list(W1=W1,W2=W2),B=list(B0=B0,B1=B1,B2=B2)))
        return(list(X=X,W=list(W1=W1,W2=W2),B=B))

      }

      # W1 = matrix(runif(D*floor(P/2),0,1),nrow=D,ncol=floor(P/2)) # weights
      # W2 = matrix(runif(D*(P-floor(P/2)),0,1),nrow=D,ncol=P-floor(P/2)) # weights
      #
      # X1 = matrix(rnorm(N*floor(P/2), 10*qrnn::sigmoid(Z2%*%W1), sd = 0.25), nrow=N, ncol=floor(P/2))
      # X2 = matrix(rnorm(N*(P-ncol(X1)), 5*(cos(Z1%*%W2) + qrnn::sigmoid(Z2%*%W2)), sd = 0.25), nrow=N, ncol=P-ncol(X1))
      #
      # X=cbind(X1,X2)

      if(P_real>0){
        fit = generate_nonlinear_data(Z=Z, H=H, type="real", P_type=P_real, C=NULL)
        X[,data_types_x=="real"] = fit$X
      }
      if(P_cat>0){
        fit = generate_nonlinear_data(Z=Z, H=H, type="cat", P_type=P_cat, C=NULL)
        X[,data_types_x=="cat"] = fit$X
      }
      if(P_count>0){
        fit = generate_nonlinear_data(Z=Z, H=H, type="count", P_type=P_count, C=NULL)
        X[,data_types_x=="count"] = fit$X
      }
      if(P_pos>0){
        fit = generate_nonlinear_data(Z=Z, H=H, type="pos", P_type=P_pos, C=NULL)
        X[,data_types_x=="pos"] = fit$X
      }

    }else{
      set.seed(seed)
      Z=matrix(nrow=N,ncol=D)
      for(d in 1:D){
        Z[,d]=rnorm(N,mean=0,sd=1)
      }

      generate_linear_data = function(Z,type=c("real","cat","count","pos"), P_type, C=NULL){
        # ### tests:
        # type="real"; P_type=5; C=NULL
        # type="cat"; P_type=1; C=3
        # type="count"; P_type=5; C=NULL
        # type="pos"; P_type=5; C=NULL
        # ############
        N = nrow(Z); D = ncol(Z)
        if(type %in% c("real", "count","pos")){
          P0 = P_type
        } else if(type =="cat"){
          P0 = C*P_type
        }
        W = if(type=="cat"){
          # matrix(3*c(-1,-1,-1,
          #            1,1,1),nrow=D,ncol=P0)
          matrix(3*sample(c(-1,1),size=D*P0,replace=T),nrow=D,ncol=P0)
          ## ord:
          # matrix(runif(D*P0,0,1),nrow=D,ncol=P0)
        }else{
          # matrix(sample(c(-1,1),D*P0,replace=T)*runif(D*P0,0.5,2),nrow=D,ncol=P0)  # unif (1,2) and +/- random
          # matrix(sample(c(-1,1),D*P0,replace=T)*runif(D*P0,0.5,1),nrow=D,ncol=P0)  # unif (1,2) and +/- random
          # matrix(rnorm(D*P0,0,1),nrow=D,ncol=P0)
          matrix(rnorm(D*P0,0,0.5),nrow=D,ncol=P0)
        }

        # B = matrix(rnorm(N*P0,0,0.25),nrow=N,ncol=P0)
        B = matrix(rnorm(N*P0,0,1),nrow=N,ncol=P0)
        # B0 = 0 #location shift?
        B0 = 2 ### maybe change this shift to 10

        eta = Z %*% W + B
        eta = apply(eta,2,function(x){(x-mean(x))/sd(x)})  # pre-normalize
        eta = eta + B0  # add location shift

        # eta = Z %*% W
        if(type=="real"){
          # X = matrix(rnorm(N*P_type, eta, 0.25), N, P_type)
          X = eta  # if B matrix (noise) is applied to eta
          # X = apply(eta, 2, function(x) (x-mean(x))/sd(x))    # make it mean 0 sd 1
        }else if(type=="cat"){
          X = matrix(nrow=N,ncol=0)
          for(j in 1:P_type){
            eta0 = eta[,(C*(j-1)+1):(C*j)]  # subset relevant C columns of linear transformation from Z
            probs = t(apply(eta0, 1, function(x) softmax(x)))
            classes = apply(eta0, 1, function(x){which.max(rmultinom(1,1,softmax(x)))}) # eta0 --> softmax probs --> multinom --> designate class
            X = cbind(X, classes)
          }

          ## ORD SIM (mimicking ordsample() from GenOrd package)
          # Sigma = diag(P_type); Sigma[Sigma==0] = 0.5
          # # eig=eigen(Sigma,symmetric=T)
          # # sqrteigval = diag(sqrt(eig$values),nrow=nrow(Sigma),ncol=ncol(Sigma))
          # # eigvec <- eig$vectors
          # # fry <- eigvec %*% sqrteigval
          # #
          # # X <- matrix(rnorm(P_type * N), N)
          # # X <- scale(X, TRUE, FALSE)
          # # X <- X %*% svd(X, nu = 0)$v
          # # X <- scale(X, FALSE, TRUE)
          # # X <- fry %*% t(X)
          # # X <- t(X)        ## https://github.com/AFialkowski/SimMultiCorrData/blob/master/R/rcorrvar.R see Y_cat?
          #
          # marginal=list(); support=list()
          # for(i in 1:P_type){
          #   marginal[[i]] = c(1:(C-1))/C
          #   support[[i]] = 1:C   # can be different Cs across variables, but keep it constant for now
          # }
          # # draw = matrix(MASS::mvrnorm(N*P_type, rep(0,P_type), Sigma), nrow=N, ncol=P_type)
          # # draw = matrix(rnorm(N*P_type, eta, Sigma), nrow=N, ncol=P_type)
          # draw = eta
          # X = matrix(nrow=N,ncol=P_type)
          # for(i in 1:P_type){
          #   X[,i] = as.integer(cut(draw[,i], breaks=c(min(draw[,i])-1, qnorm(marginal[[i]]), max(draw[,i])+1)))
          #   X[,i] = support[[i]][X[,i]]
          # }

        }else if(type=="count"){
          X = matrix(rpois(N*P_type, eta+100), N, P_type)  # add constant of 100 to lambda --> larger counts
        }else if(type=="pos"){
          X = matrix(rlnorm(N*P_type, eta, 0.1), N, P_type) # log-sd changed to have range of about 0 - 80
        }
        return(list(X=X,W=W,B=list(B0=B0,B=B)))
      }

      if(P_real>0){
        fit = generate_linear_data(Z=Z, type="real", P_type=P_real, C=NULL)
        X[,data_types_x=="real"] = fit$X; Ws$"real" = fit$W; Bs$"real" = fit$B }
      if(P_cat>0){
        fit = generate_linear_data(Z=Z, type="cat", P_type=P_cat, C=C)
        X[,data_types_x=="cat"] = fit$X; Ws$"cat" = fit$W; Bs$"cat" = fit$B }
      if(P_count>0){
        fit = generate_linear_data(Z=Z, type="count", P_type=P_count, C=NULL)
        X[,data_types_x=="count"] = fit$X; Ws$"count" = fit$W; Bs$"count" = fit$B}
      if(P_pos>0){
        fit = generate_linear_data(Z=Z, type="pos", P_type=P_pos, C=NULL)
        X[,data_types_x=="pos"] = fit$X; Ws$"pos" = fit$W; Bs$"pos" = fit$B}

    }


    if(family=="Multinomial"){
      beta1 = matrix(0, ncol=1, nrow=P_real+P_count+P_cat)  # class 1: reference. effects of X on Pr(Y) are 0
      betas_real = matrix(sample(c(-1*beta, beta), P_real*(Cy-1), replace=T),nrow=P_real,ncol=Cy-1)
      # (most proper:  for cat vars: diff effect per level. not doing this right now --> same effect from 1 --> 2, 2 --> 3, etc.)
      # betas_count = sample(c(-beta/600, beta/600), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
      # betas_cat = sample(c(-2*beta, 2*beta), P_cat, replace=T)       # mean(cts) = 5, mean(cat) = 2, mean(count) = exp(8) ~ 2981
      betas_count = matrix(sample(c(-1*beta, beta), P_count*(Cy-1), replace=T),nrow=P_count,ncol=Cy-1)      # effect is twice the magnitude of the effect of continuous values?

      betas_cat = matrix(sample(c(-2*beta, 2*beta), P_cat*(Cy-1), replace=T), nrow=P_cat, ncol=Cy-1)
      betas = cbind(beta1, rbind(betas_real, betas_cat, betas_count))
      epsilon = matrix(rnorm(N*(Cy-1), 0, 0.1), nrow=N,ncol=Cy)


      betas2_real = matrix(sample(c(-1*beta, beta), P_real*(Cy-1), replace=T),nrow=P_real,ncol=Cy-1)
      betas2_count = matrix(sample(c(-1*beta, beta), P_count*(Cy-1), replace=T),nrow=P_count,ncol=Cy-1)      # effect is twice the magnitude of the effect of continuous values?
      betas2_cat = matrix(sample(c(-2*beta, 2*beta), P_cat*(Cy-1), replace=T), nrow=P_cat, ncol=Cy-1)

      betas2 = cbind(beta1, rbind(betas2_real, betas2_cat, betas2_count))

    } else{
      betas_real = sample(c(-1*beta, beta), P_real, replace=T)
      # (most proper:  for cat vars: diff effect per level. not doing this right now --> same effect from 1 --> 2, 2 --> 3, etc.)
      # betas_count = sample(c(-beta/600, beta/600), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
      # betas_cat = sample(c(-2*beta, 2*beta), P_cat, replace=T)       # mean(cts) = 5, mean(cat) = 2, mean(count) = exp(8) ~ 2981
      betas_count = sample(c(-1*beta, beta), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
      betas_cat = sample(c(-2*beta, 2*beta), P_cat, replace=T)
      betas = c(betas_real, betas_cat, betas_count)
      epsilon = rnorm(N, 0, 0.1)

      betas2_real = sample(c(-4*beta, beta), P_real, replace=T)
      betas2_count = sample(c(-4*beta, beta), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
      betas2_cat = sample(c(-8*beta, 8*beta), P_cat, replace=T)
      betas2 = c(betas2_real, betas2_cat, betas2_count)
    } # mean(cts) = 5, mean(cat) = 2, mean(count) = exp(8) ~ 2981

    # epsilon = 0
    if(NL_y){
      ##### covariates of Y are X^2
      if(family=="Gaussian"){
        # Simulate y from X --> y = Xb + e
        beta0s = 0
        # eta = beta0s + X^2 %*% betas + epsilon
        eta = beta0s + X^2 %*% betas + log(X + abs(min(X)) + 0.001) %*% betas2 + epsilon
        Y = eta
        prs = NA
      } else if(family=="Multinomial"){
        beta0s = 0
        # eta = beta0s + X^2 %*% betas + epsilon
        eta = beta0s + X^2 %*% betas + log(X + abs(min(X)) + 0.001) %*% betas2 + epsilon
        prs = t(apply(eta,1,softmax))
        Y = apply(prs, 1, function(x) which.max(rmultinom(1, 1, x)))  # categorical distribution

        # Y = rbinom(N,1,prs)  # for binary outcome only
      } else if(family=="Poisson"){
        beta0s = 8
        # eta = beta0s + X^2 %*% betas + epsilon
        eta = beta0s + X^2 %*% betas + log(X + abs(min(X)) + 0.001) %*% betas2 + epsilon
        Y = round(exp(eta),0)   # log(Y) = eta. Round Y to integer (to simulate count)
        prs = NA
      }
    }else{
      ##### covariates for Y are X
      # family="Gaussian" --> Gaussian data for Y
      if(family=="Gaussian"){
        # Simulate y from X --> y = Xb + e
        beta0s = 0
        eta = beta0s + X %*% betas + epsilon
        # betas = sample(c(-1*beta,beta), P, replace=T)   # defined together in mixed data types
        Y = eta
        prs = NA
      } else if(family=="Multinomial"){

        ### I THINK THIS WORKS FOR LOGISTIC REGRESSION, NOT FOR MULTINOMIAL
        find_int = function(p,betas) {
          # Define a path through parameter space
          f = function(t){
            sapply(t, function(y) mean(1 / (1 + exp(-y -as.matrix(X) %*% betas))))
          }
          alpha <- uniroot(function(t) f(t) - p, c(-1e6, 1e6), tol = .Machine$double.eps^0.5)$root
          return(alpha)
        }
        beta0s <- sapply(c(0.5), function(y)find_int(y,betas[,2]))

        eta = beta0s + X %*% betas + epsilon

        inv_logit = function(x){
          return(1/(1+exp(-x)))
        }
        prs = inv_logit(eta[,2])
        Y = rbinom(N, 1, prs)  # for binary outcome only
        ## transform prs into what it should have been (reference: 1st class)
        prs0 = cbind(1-prs, prs)
        prs = prs0

        #### MULTINOMIAL. NEED TO FIX beta0s TO HAVE CLASSES ~ N/C expected per class (see above for LR). NEED TO CHANGE EVENTUALLY.
        # beta0s = 0     # makes class too unbalanced...
        # eta = beta0s + X %*% betas + epsilon
        # prs = t(apply(eta,1,softmax))
        # Y = apply(prs, 1, function(x) which.max(rmultinom(1, 1, x)))  # categorical distribution

      } else if(family=="Poisson"){
        beta0s = 8
        eta = beta0s + X %*% betas + epsilon
        # betas = sample(c(-1*beta,beta), P, replace=T)    # defined together in mixed data types
        Y = round(exp(eta),0)   # log(Y) = eta. Round Y to integer (to simulate count)
        prs = NA
      }
    }
    Y = matrix(Y,ncol=1)

    # hist(Y)

    # X[,data_types_x=="count"] = exp(X[,data_types_x=="count"])    # undo log transf of X

    params = list(beta0s=beta0s, betas=betas, betas2=betas2, beta=beta, prs = prs, Ws=Ws, Bs=Bs)  # betas2 only used as a second covariate in NL_y (experimental)
  }
  data = list(X=X, Y=Y)
  return(list(data=data, params=params))
}

#' simulateMask: Simulates missingness mask on X, based on specified mechanism and other parameters, like phi
#'
#' @param data Data matrix X used as covariates for simulating the missingness
#' @param scheme UV (default) or MV: Univariate or multivariate missingness model
#' @param mechanism Mechanism of missingness ("MCAR", "MAR", or "MNAR")
#' @param NL_r Boolean: nonlinear generation of R from covariate(s)
#' @param pi Proportion of missingness in each partially-observed column, ID'ed by miss_cols
#' @param phis Coefficient of missingness model for each feature in X
#' @param miss_cols Indices of columns containing missingness (partially-observed features in X)
#' @param ref_cols Indices of columns not containing missingness (fully-observed features in X)
#' @param seed Seed to generate reproducible data
#' @return list with Missing: simulated mask of X, probs: probability of each entry in X being observed, params: phis, miss_cols, ref_cols, scheme
#' @examples
#' EXAMPLE HERE
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/dlglm}
#'
#' @export
simulateMask = function(data, scheme, mechanism, NL_r, pi, phis, miss_cols, ref_cols, seed){

  # mechanism of missingness
  # 1 - pi: % of missingness
  # miss_cols: columns to induce missingness on
  # ref_cols: (FOR MAR) columns that are fully observed used as covariate
  # seed determined by sim_index
  n=nrow(data)
  # function s.t. expected proportion of nonmissing (Missing=1) is p. let p=1-p_miss
  find_int = function(p,beta) {
    # Define a path through parameter space
    f = function(t){
      sapply(t, function(y) mean(1 / (1 + exp(-y -x %*% phi))))
    }
    alpha <- uniroot(function(t) f(t) - p, c(-1e6, 1e6), tol = .Machine$double.eps^0.5)$root
    return(alpha)
  }
  logit = function(x){
    if(x<=0 | x>=1){stop('x must be in (0,1)')}
    return(log(x/(1-x)))
  }

  Missing = matrix(1,nrow=nrow(data),ncol=ncol(data))
  prob_Missing = matrix(1,nrow=nrow(data),ncol=ncol(data))
  params=list()

  set.seed(seed)  # for reproducibility

  for(j in 1:length(miss_cols)){
    # specifying missingness model covariates
    if(mechanism=="MCAR"){
      x <- matrix(rep(0,n),ncol=1) # for MCAR: no covariate (same as having 0 for all samples)
      phi=0                       # for MCAR: no effect of covariates on missingness (x is 0 so phi doesn't matter)
    }else if(mechanism=="MAR"){
      if(scheme=="UV"){
        covar = if(NL_r){ log(data[,ref_cols[j]] + abs(data[,ref_cols[j]]) + 0.01) }else{ data[,ref_cols[j]] }
        x <- matrix(covar,ncol=1)             # missingness dep on just the corresponding ref column (randomly paired)
        phi=phis[ref_cols[j]]
      }else if(scheme=="MV"){
        # check if missing column in ref. col: this would be MNAR (stop computation)
        if(any(ref_cols %in% miss_cols)){stop(sprintf("missing cols in reference. is this intended? this is MNAR not MAR."))}
        covars = if(NL_r){ apply(data[,ref_cols], 2, function(x){log(x+abs(x)+0.01)}) }else{ data[,ref_cols] }
        x <- matrix(covars,ncol=length(ref_cols)) # missingness dep on all ref columns
        phi=phis[ref_cols]
      }
    }else if(mechanism=="MNAR"){
        # Selection Model
        if(scheme=="UV"){
          # MISSINGNESS OF EACH MISS COL IS ITS OWN PREDICTOR
          covar = if(NL_r){ log(data[,miss_cols[j]] + abs(data[,miss_cols[j]]) + 0.01) }else{ data[,miss_cols[j]] }
          x <- matrix(covar,ncol=1) # just the corresponding ref column
          phi=phis[miss_cols[j]]
        }else if(scheme=="MV"){
          # check if missing column not in ref col. this might be MAR if missingness not dep on any other missing data
          if(all(!(ref_cols %in% miss_cols))){warning(sprintf("no missing cols in reference. is this intended? this might be MAR not MNAR"))}
          covars = if(NL_r){ apply(data[,ref_cols], 2, function(x){log(x+abs(x)+0.01)}) }else{ data[,ref_cols] }
          x <- matrix(data[,ref_cols],ncol=length(ref_cols)) # all ref columns
          phi=phis[ref_cols]         # in MNAR --> ref_cols can overlap with miss_cols (dependent on missingness)

          # address when miss_cols/ref_cols/phis are not null (i.e. want to induce missingness on col 2 & 5 based on cols 1, 3, & 4)
        }
    }
    alph <- sapply(1-pi, function(y) find_int(y,phi))
    mod = alph + x%*%phi

    # Pr(Missing_i = 1)
    probs = inv_logit(mod)
    prob_Missing[,miss_cols[j]] = probs

    # set seed for each column for reproducibility, but still different across columns
    Missing[,miss_cols[j]] = rbinom(n,1,probs)

    params[[j]]=list(phi0=alph, phi=phi, miss=miss_cols[j], ref=ref_cols[j], scheme=scheme)
    print(c(mean(data[Missing[,miss_cols[j]]==0,miss_cols[j]]), mean(data[Missing[,miss_cols[j]]==1,miss_cols[j]])))
  }

  return(list(Missing=Missing,
              probs=prob_Missing,
              params=params))
}

#' prepareData: Main simulation wrapper function
#'
#' @param dataset Dataset name (default: "SIM"). Inference and prediction L1 computed just for SIM
#' @param data.file.name Default is NULL. Can load custom data file name with sim.data, X, and Y
#' @param mask.file.name Default is NULL. Can load custom missingness file name with sim.mask, mask_x, and mask_y
#' @param sim.params List of simulation parameters. Contains N, D, P, data_types_x, family, sim_index, ratios (split of train, valid, test)
#' @param miss.params List of missingness parameters. Contains scheme, mechanism, pi, phi0, miss_pct_features, miss_cols, ref_cols, NL_r
#' @param case Simulation case of missingness (missingness in x, y, or both xy)
#' @return Saves and returns list of simulation parameters and data: X, Y, mask_x, mask_y, g, sim.params, miss.params, sim.data, sim.mas
#' @examples
#' EXAMPLE HERE
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/dlglm}
#'
#' @export
prepareData = function(dataset = "SIM", data.file.name = NULL, mask.file.name=NULL,
                       sim.params = list(N=1e5, D=2, P=8, data_types_x=NA, family="Gaussian", sim_index=1, ratios=c(train=.8,valid=.1,test=.1),
                                         beta=.25, C=NULL, Cy=NULL, NL_x=F, NL_y=F),
                       miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_pct_features=50, miss_cols=NULL, ref_cols=NULL, NL_r=F),
                       case=c("x","y","xy")){
  print(sim.params)

  if(grepl("SIM",dataset)){
    data_type_x = if(all(sim.params$data_types_x==sim.params$data_types_x[1])){ sim.params$data_types_x[1] } else{ "mixed" }
    data_type_y = if(sim.params$family=="Gaussian"){"real"}else if(sim.params$family=="Multinomial"){"cat"}else if(sim.params$family=="Poisson"){"cts"}

    iNL_x = if(sim.params$NL_x){"NL"}else{""}
    iNL_y = if(sim.params$NL_y){"NL"}else{""}
    iNL_r = if(miss.params$NL_r){"NL_"}else{""}
    dir_name = sprintf("Results_%sX%s_%sY%s/%s/miss_%s%s/phi%d/sim%d", iNL_x, data_type_x, iNL_y, data_type_y, dataset, iNL_r, case, miss.params$phi0, sim.params$sim_index)
  }else{
    iNL_r = if(miss.params$NL_r){"NL_"}else{""}
    dir_name = sprintf("Results_%s/miss_%s%s/phi%d/sim%d", dataset, iNL_r, case, miss.params$phi0, sim.params$sim_index)
  }
  diag_dir_name = sprintf("%s/Diagnostics", dir_name)
  ifelse(!dir.exists(dir_name), dir.create(dir_name,recursive=T), F)
  ifelse(!dir.exists(diag_dir_name), dir.create(diag_dir_name,recursive=T), F)

  # data (simulate or read)
  if(!is.null(data.file.name)){
    # read existing data. list object "data" should be there, with entries "X" and "Y"
    load(data.file.name)
    dataset = unlist(strsplit(data.file.name,"[.]"))[1]   # remove extension
  }else{
    if(grepl("SIM",dataset)){
      P=sim.params$P; N=sim.params$N
      if(all(is.na(sim.params$data_types_x))){sim.params$data_types_x = rep("real",sim.params$P)}
      dataset = sprintf("Xr%dct%dcat%d_beta%f_pi%d/SIM_N%d_P%d_D%d",sum(sim.params$data_types_x=="real"),
                        sum(sim.params$data_types_x=="count"),sum(sim.params$data_types_x=="cat"), params$beta, miss.params$pi*100,
                        sim.params$N, sim.params$P, sim.params$D)
    }
    sim.data = simulateData(dataset=dataset, N=sim.params$N, D=sim.params$D, P=sim.params$P,
                            data_types_x=sim.params$data_types_x,
                            family=sim.params$family, NL_x=sim.params$NL_x, NL_y=sim.params$NL_y,
                            seed=sim.params$sim_index*9,
                            beta=sim.params$beta, C=sim.params$C, Cy=sim.params$Cy)
    params = sim.data$params

    # family = "Gaussian", "Multinomial", or "Poisson"
    data = sim.data$data
    X=data$X; Y=data$Y
    N = nrow(X); P = ncol(X)
  }

  print("Data simulated")
  miss_pct_features = miss.params$miss_pct_features

  # mask (simulate or read)
  if(!is.null(mask.file.name)){
    # read existing mask. object "mask" should be there
    load(mask.file.name)
    phis=NULL; miss_cols=NULL; ref_cols=NULL
    mechanism = NA; pi=NA
  }else{
    mechanism = miss.params$mechanism; pi = miss.params$pi
    set.seed( sim.params$sim_index*9 )
    # phis = c(0, rlnorm(P, log(miss.params$phi0), 0.2)) # default: 0 for intercept, draw values of coefs from log-norm(5, 0.2)
    phis = rlnorm(P, log(miss.params$phi0), 0.2) # SHOULDN"T CONTAIN INTERCEPT... intercept found by critical point in E[R] ~ pi
    if(grepl("SIM",dataset)){  if(any(sim.params$data_types_x=="cat")){ phis[sim.params$data_types_x=="cat"] = phis[sim.params$data_types_x=="cat"] * 2 }  } # categorical effects on missingness is 1/5th of real effects
    # if no reference/miss cols provided --> randomly select 50% each
    if(is.null(miss.params$miss_cols) & is.null(miss.params$ref_cols)){
      # miss_cols = sample(1:P, floor(P/2), F); ref_cols = c(1:P)[-miss_cols]
      miss_pct_features = miss.params$miss_pct_features
      miss_cols = sample(1:P, floor(P*miss_pct_features/100), F)
      ref_cols = c(1:P)[-miss_cols]
    }else if(!is.null(miss.params$miss_cols) & !is.null(miss.params$ref_cols)){
      miss_cols = miss.params$miss_cols; ref_cols = miss.params$ref_cols
    }else if(is.null(miss.params$miss_cols)){
      ref_cols = miss.params$ref_cols; miss_cols = c(1:P)[-ref_cols]
    }else if(is.null(miss.params$ref_cols)){
      miss_cols = miss.params$miss_cols; ref_cols = c(1:P)[-miss_cols]
    }

    if(case=="x"){
      data=X
      ref_cols0=ref_cols; miss_cols0=miss_cols
    }else if(case=="y"){
      data=data.frame(cbind(X[,ref_cols[1]],Y))
      ref_cols0=1; miss_cols0=2
    }else if(case=="xy"){
      data=data.frame(cbind(X,Y))
      miss_cols0=c(miss_cols,  ncol(data))  # add Y to the columns where missingness is imposed
      ref_cols0 = ref_cols
    }
    print(miss.params$scheme); print(miss.params$mechanism); print(miss.params$NL_r); print(miss.params$pi); print(phis)
    print(miss_cols0); print(ref_cols0); print(sim.params$sim_index)
    sim.mask = simulateMask(data=data, scheme=miss.params$scheme, mechanism=miss.params$mechanism, NL_r=miss.params$NL_r,
                            pi=miss.params$pi, phis=phis, miss_cols=miss_cols0, ref_cols=ref_cols0,
                            seed = sim.params$sim_index*9)
    mask = sim.mask$Missing    # 1 observed, 0 missing
    if(case=="x"){
      mask_x = mask; mask_y = rep(1, N)
    }else if(case=="y"){
      mask_x = matrix(1, nrow=N, ncol=P); mask_y = mask
    }else if(case=="xy"){
      mask_x = mask[,1:P]; mask_y = mask[,(P+1)]
    }
  }

  print("Mask simulated")

  if(grepl("x",case)){
    # plot missing X vals (1 col)
    for(i in 1:length(miss_cols)){
      png(sprintf("%s/%s_col%d_X.png",diag_dir_name, mechanism, miss_cols[i]))
      boxplot(X[mask_x[,miss_cols[i]]==0,miss_cols[i]],
              X[mask_x[,miss_cols[i]]==1,miss_cols[i]],
              X[mask_x[,miss_cols[i]]==0,ref_cols[i]],
              X[mask_x[,miss_cols[i]]==1,ref_cols[i]],
              names =c(sprintf("Missing, col%d",miss_cols[i]), sprintf("Observed, col%d",miss_cols[i]),
                       sprintf("Missing, col%d",ref_cols[i]), sprintf("Observed, col%d",ref_cols[i])),
              main=sprintf("Values of miss_col%d and corr ref_col%d",miss_cols[i],ref_cols[i]),
              sub="Stratified by mask in miss_col",outline=F)
      dev.off()
    }
  }
  if(grepl("y",case)){
    png(sprintf("%s/%s_Y.png",diag_dir_name, mechanism))
    # plot missing Y vals
    boxplot(Y[mask_y==0,],
            Y[mask_y==1,],
            X[mask_y==0,ref_cols[1]],
            X[mask_y==1,ref_cols[1]],
            names =c("Masked, Y", "Unmasked, Y", "Masked, ref_col", "Unmasked, ref_col"),
            main="Values of Y and 1st ref_col (in X)", sub="Stratified by mask in Y",outline=F)
    dev.off()

  }

  # save: X, Y, mask_x, mask_y
  # save extra params:

  miss.params$miss_cols = miss_cols; miss.params$ref_cols = ref_cols; miss.params$phis = phis

  g = sample(cut(
    seq(nrow(data)),
    nrow(data)*cumsum(c(0,sim.params$ratios)),
    labels = names(sim.params$ratios)
  ))

  sim.data$data = NULL; sim.mask$Missing = NULL # these have already been extracted. don't save to params file

  save(list=c("X","Y","mask_x","mask_y","g"), file = sprintf("%s/data_%s_%d_%d.RData", dir_name, mechanism, miss.params$miss_pct_features, pi*100))
  save(list=c("sim.params","miss.params","sim.data","sim.mask"), file = sprintf("%s/params_%s_%d_%d.RData", dir_name, mechanism, miss.params$miss_pct_features, pi*100))
  print("X:")
  print(head(X,n=20))
  print("Y:")
  print(head(Y,n=20))
  print("mask_x")
  print(head(mask_x,n=20))
  print("mask_y")
  print(head(mask_y,n=20))
  return(list(X=X, Y=Y, mask_x=mask_x, mask_y=mask_y, g=g, sim.params=sim.params, miss.params=miss.params,
              sim.data=sim.data, sim.mask=sim.mask))
}
