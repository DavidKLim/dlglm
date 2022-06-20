#' processResults: Processes final dlglm, idlglm, and mice results
#'
#' @param dataset Dataset name (default: "SIM"). Inference and prediction L1 computed just for SIM
#' @param prefix Prefix for dataset (default: "").
#' @param data.file.name Default is NULL. Can load custom data file name with sim.data, X, and Y
#' @param mask.file.name Default is NULL. Can load custom missingness file name with sim.mask, mask_x, and mask_y
#' @param sim.params List of simulation parameters. Contains N, D, P, data_types_x, family, sim_index, ratios (split of train, valid, test)
#' @param miss.params List of missingness parameters. Contains scheme, mechanism, pi, phi0, miss_pct_features, miss_cols, ref_cols, NL_r
#' @param case Simulation case of missingness (missingness in x, y, or both xy)
#' @param data_types_x Vector of data types ('real', 'count', 'pos', 'cat')
#' @return res object: with imputation ("imp"), inference ("inf"), and prediction ("pred") metrics
#' @examples
#' EXAMPLE HERE
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/dlglm}
#'
#' @importFrom reticulate import
#' @importFrom nnet multinom
#' @importFrom pROC auc
#' @importFrom measures Brier KAPPA F1 FPR TPR PPV
#' @importFrom gridExtra tableGrob
#' @importFrom grid textGrob grobHeight
#' @importFrom gtable gtable_add_rows gtable_add_grob
#' @importFrom mclust adjustedRandIndex
#' @importFrom mice complete
#' @importFrom yardstick mcc_vec
#' @import ggplot2
#'
#' @export
### processes results (imputation, inference, prediction) for data where missingness is simulated
processResults = function(dataset="SIM",prefix="",data.file.name = NULL, mask.file.name=NULL,
                          sim.params = list(N=1e5, D=2, P=8, family="Gaussian", sim_index=1, ratios=c(train=.6,valid=.2,test=.2),
                                            mu=0, sd=1, beta=5, C=3, Cy=NULL, NL_x=F, NL_y=F),
                          miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_pct_features=50, miss_cols=NULL, ref_cols=NULL, NL_r=F),
                          case=c("x","y","xy"), data_types_x, data_type_y = "real", family="Gaussian", methods=c("idlglm","dlglm","mice","zero","mean"), init_r="default", normalize=F){
  inorm = if(normalize){"with_normalization/"}else{""}
  torch = import("torch")

  if(init_r=="default"){dlglm_pref = ""} else if(init_r=="alt"){dlglm_pref = "/alt_init"}

  inv_logit = function(x){
    return(1/(1+exp(-x)))
  }  # can take this out once dlglm is packaged
  logsumexp <- function (x) {
    y = max(x)
    y + log(sum(exp(x - y)))
  }
  softmax <- function (x) {
    exp(x - logsumexp(x))
  }
  if(grepl("SIM",dataset)){
    N=sim.params$N; P=sim.params$P; D=sim.params$D
    if(is.null(data.file.name)){
      dataset = sprintf("SIM_N%d_P%d_D%d", N, P, D)
    } else{dataset = unlist(strsplit(data.file.name,"[.]"))[1]}
  } else{ D = NA }
  family=sim.params$family
  link=if(family=="Gaussian"){"identity"}else if(family=="Multinomial"){"mlogit"}else if(family=="Poisson"){"log"}

  pi = miss.params$pi
  mechanism=miss.params$mechanism
  sim_index=sim.params$sim_index

  iNL_r=""
  if(grepl("SIM",dataset)){
    iNL_x = if(NL_x){"NL"}else{""}
    iNL_y = if(NL_y){"NL"}else{""}
    data_type_x = if(all(data_types_x==data_types_x[1])){data_types_x[1]}else{"mixed"}

    dir_name0 = sprintf("Results_%sX%s_%sY%s/%s%s/miss_%s%s/phi%d/sim%d", iNL_x,data_type_x,iNL_y,data_type_y,prefix, dataset, iNL_r, case, miss.params$phi0, sim_index)
    dir_name=sprintf("%s%s",dir_name0,dlglm_pref)
  } else{
    dir_name0 = sprintf("Results_%s%s/miss_%s%s/phi%d/sim%d", prefix,dataset, iNL_r, case, miss.params$phi0, sim_index)
    dir_name=sprintf("%s%s",dir_name0,dlglm_pref)
  }
  # to save interim results
  # if(Ignorable){dir_name = sprintf("%s/Ignorable",dir_name0)}else{dir_name=dir_name0}
  ifelse(!dir.exists(sprintf("%s/Diagnostics",dir_name)), dir.create(sprintf("%s/Diagnostics",dir_name)), F)
  ifelse(!dir.exists(sprintf("%s/%s_%d_%d",dir_name, mechanism, miss.params$miss_pct_features, pi*100)),
         dir.create(sprintf("%s/%s_%d_%d",dir_name, mechanism, miss.params$miss_pct_features, pi*100)), F)

  data.fname = sprintf("%s/data_%s_%d_%d.RData", dir_name0, mechanism, miss_pct_features, pi*100)
  print(paste("Data file: ", data.fname))
  if(!file.exists(data.fname)){
    stop("Data file does not exist..")
  } else{
    load(data.fname)
  }

  N=nrow(X); P=ncol(X)
  if(!grepl("SIM",dataset) & family=="Multinomial"){ levels_Y = levels(factor(Y)); Y = as.numeric(as.factor(Y)) }  # sometimes the cat vars are nonnumeric for UCI

  data_types_x_0 = data_types_x
  # mask_x = (res$mask_x)^2; mask_y = (res$mask_y)^2
  if(sum(data_types_x=="cat") == 0){
    X_aug = X
    # mask_x_aug = mask_x
  } else{
    # reorder to real&count covariates first, then augment cat dummy vars
    X_aug = X[,!(data_types_x %in% c("cat"))]
    # mask_x_aug = mask_x[,!(data_types_x %in% c("cat"))]

    ## onehot encode categorical variables
    # X_cats = X[,data_types_x=="cat"]
    Cs = rep(0,sum(data_types_x=="cat"))
    # X_cats_onehot = matrix(nrow=N,ncol=0)
    cat_ids = which(data_types_x=="cat")
    for(i in 1:length(cat_ids)){
      X_cat = as.numeric(as.factor(X[,cat_ids[i]]))-1
      Cs[i] = length(unique(X_cat))
      X_cat_onehot = matrix(ncol = Cs[i], nrow=length(X_cat))
      for(ii in 1:Cs[i]){
        X_cat_onehot[,ii] = (X_cat==ii-1)^2
      }
      # X_cats_onehot = cbind(X_cats_onehot, X_cat_onehot)
      X_aug = cbind(X_aug, X_cat_onehot)
      # mask_x_aug = cbind(mask_x_aug, matrix(mask_x[,cat_ids[i]], nrow=N, ncol=Cs[i]))
    }


    ## column bind real/count and one-hot encoded cat vars
    data_types_x = c( data_types_x[!(data_types_x %in% c("cat"))], rep("cat",sum(Cs)) )
  }

  Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Xs_aug = split(data.frame(X_aug), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Rxs = split(data.frame(mask_x), g)
  Rys = split(data.frame(mask_y), g)

  Cy = length(unique(Ys$test$Y))
  if(normalize){
    norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))
    norm_mean_y=0; norm_sd_y=1
    norm_means_x[data_types_x=="cat"] = 0; norm_sds_x[data_types_x=="cat"] = 1
    # if(family=="Multinomial"){
    #   norm_mean_y=0; norm_sd_y=1
    # } else{
    #   norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
    # }
    # dir_name = sprintf("%s/with_normalization",dir_name)
    # ifelse(!dir.exists(dir_name),dir.create(dir_name),F)
  }else{
    P_aug = ncol(Xs$train)
    norm_means_x = rep(0, P_aug); norm_sds_x = rep(1, P_aug)
    norm_mean_y = 0; norm_sd_y = 1
  }
  ## didn't normalize
  # norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))   # normalization already undone in results xhat
  # # norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
  # norm_mean_y=0; norm_sd_y=1   # didn't normalize Y

  miss_cols = which(colMeans(mask_x)!=1)

  tab = matrix(nrow = P, ncol=0)
  xhats = list()
  yhats = matrix(ncol=0,nrow=nrow(Ys$test))
  yhats2 = matrix(ncol=0,nrow=nrow(Ys$test))
  prhats = list()
  prhats2 = list()    # with missingness in test set (preimpute separately for mean/zero/mice, dlglm can take it in and sample from posterior)
  prhats_train = list()

  if(family=="Multinomial"){ link = "mlogit"
  } else if(family=="Gaussian"){ link = "identity"
  } else if(family=="Poisson"){ link = "log"
  } else if(family=="Binomial"){ link = "logit" }

  invlink = function(link){
    if(link=="identity"){fx = torch$nn$Identity(0L)
    } else if(link=="log"){fx = torch$exp
    } else if(link=="logit"){fx = torch$nn$Sigmoid()
    } else if(link=="mlogit"){fx = torch$nn$Softmax(dim=1L)}
    return(fx)
  }

  ## Process mean/zero imputation of X and Y ##
  if("zero" %in% methods){
    X_zero = X; X_zero[mask_x==0]=0
    Y_zero = Y; Y_zero[mask_y==0]=0
    Xs_zero = split(data.frame(X_zero), g)
    Ys_zero = split(data.frame(Y_zero), g)
    xhat_zero = Xs_zero$test; yhat_zero = Ys_zero$test

    if(family=="Gaussian"){
      fit_zero = glm(Y_zero ~ 0 + . , data=cbind( Xs_zero$train, Ys_zero$train ))
      w_zero = c(fit_zero$coefficients)
    }else if(family=="Multinomial"){
      fit_zero = nnet::multinom(Y_zero ~ 0 + ., data=cbind( Xs_zero$train, Ys_zero$train ))
      w_zero = c(coefficients(fit_zero))
    }
    yhat_zero_pred = predict(fit_zero, newdata=cbind(Xs$test,Ys$test,row.names = NULL))
    yhat_zero_pred2 = predict(fit_zero, newdata=cbind(Xs_zero$test,Ys$test,row.names = NULL))
    yhats = cbind(yhats,yhat_zero_pred); colnames(yhats)[ncol(yhats)] = "zero"
    yhats2 = cbind(yhats2,yhat_zero_pred2); colnames(yhats2)[ncol(yhats2)] = "zero"
    if(family=="Multinomial"){
      prhat_zero_pred = predict(fit_zero, newdata=cbind(Xs$test,Ys$test,row.names = NULL), type="probs")
      prhat_zero_pred2 = predict(fit_zero, newdata=cbind(Xs_zero$test,Ys$test,row.names = NULL), type="probs")
      # if(Cy==2){
      #   prhats$"zero" = matrix(nrow=length(prhat_zero_pred),ncol=2); prhats$"zero"[,2] = prhat_zero_pred; prhats$"zero"[,1] = 1-prhats$"zero"[,2]
      #   prhats2$"zero" = matrix(nrow=length(prhat_zero_pred2),ncol=2); prhats2$"zero"[,2] = prhat_zero_pred2; prhats2$"zero"[,1] = 1-prhats2$"zero"[,2]
      # }else{
      #   prhats$"zero" = prhat_zero_pred
      #   prhats2$"zero" = prhat_zero_pred2
      # }
      prhats$"zero" = prhat_zero_pred
      prhats2$"zero" = prhat_zero_pred2
      prhats_train$"zero" = predict(fit_zero, newdata=cbind(Xs_zero$train,Ys$train,row.names = NULL), type="probs")
    }
    xhats$"zero" = xhat_zero

    tab = cbind(tab,w_zero); colnames(tab)[ncol(tab)] = "zero"
    rm(X_zero); rm(Y_zero)
    rm(Xs_zero); rm(Ys_zero)
  }
  ## created: Xs_zero (Y), yhats$zero (complete test), yhats2$zero (imputed test)
  ## prhats$zero, prhats2$zero (complete/imputed test prob of Y), xhats$zero (just test set of Xs_zero)
  gc()
  if("mean" %in% methods){
    X_mean = X; Y_mean = Y
    for(i in 1:ncol(X_mean)){
      X_mean[mask_x[,i]==0,i] = mean(X[mask_x[,i]==1,i])
    }
    Y_mean[mask_y==0] = mean(Y[mask_y==1])
    Xs_mean = split(data.frame(X_mean), g)
    Ys_mean = split(data.frame(Y_mean), g)
    xhat_mean = Xs_mean$test; yhat_mean = Ys_mean$test

    if(family=="Gaussian"){
      fit_mean = glm(Y_mean ~ 0 + . , data=cbind( Xs_mean$train ,Ys_mean$train ))
      w_mean = c(fit_mean$coefficients)
    }else if(family=="Multinomial"){
      fit_mean = nnet::multinom(as.factor(Y_mean) ~ 0+., data=cbind( Xs_mean$train ,Ys_mean$train ))
      w_mean = c(coefficients(fit_mean))
    }
    yhat_mean_pred = predict(fit_mean, newdata=cbind(Xs$test,Ys$test,row.names = NULL))
    yhat_mean_pred2 = predict(fit_mean, newdata=cbind(Xs_mean$test,Ys$test,row.names = NULL))
    yhats = cbind(yhats,yhat_mean_pred); colnames(yhats)[ncol(yhats)] = "mean"
    yhats2 = cbind(yhats2,yhat_mean_pred2); colnames(yhats2)[ncol(yhats2)] = "mean"
    if(family=="Multinomial"){
      prhat_mean_pred = predict(fit_mean, newdata=cbind(Xs$test,Ys$test,row.names = NULL), type="probs")
      prhat_mean_pred2 = predict(fit_mean, newdata=cbind(Xs_mean$test,Ys$test,row.names = NULL), type="probs")
      # if(Cy==2){
      #   prhats$"mean" = matrix(nrow=length(prhat_mean_pred),ncol=2); prhats$"mean"[,2] = prhat_mean_pred; prhats$"mean"[,1] = 1-prhats$"mean"[,2]
      #   prhats2$"mean" = matrix(nrow=length(prhat_mean_pred2),ncol=2); prhats2$"mean"[,2] = prhat_mean_pred2; prhats2$"mean"[,1] = 1-prhats2$"mean"[,2]
      # }else{
      #   prhats$"mean" = prhat_mean_pred
      #   prhats2$"mean" = prhat_mean_pred2
      # }
      prhats$"mean" = prhat_mean_pred
      prhats2$"mean" = prhat_mean_pred2
      prhats_train$"mean" = predict(fit_mean, newdata=cbind(Xs_mean$train,Ys$train,row.names = NULL), type="probs")

    }
    xhats$"mean" = xhat_mean

    tab = cbind(tab,w_mean); colnames(tab)[ncol(tab)] = "mean"
    rm(X_mean); rm(Y_mean)
    rm(Xs_mean); rm(Ys_mean)
  }
  gc()
  ## for idlglm, dlglm, mice --> load saved results
  if("dlglm" %in% methods){
    fname = sprintf("%s/res_dlglm_%s_%d_%d.RData",dir_name,mechanism,miss_pct_features,pi*100)
    print(paste("dlglm Results file: ", fname))
    if(!file.exists(fname)){ break }
    load( fname )  # loads "X","Y","mask_x","mask_y","g"
    # should contain "res", which is a list that contains "results" and "fixed.params" from now on. First iteration only contains res (no list) results object
    res = res$results
    niws = res$train_params$niws_z

    print("Loading saved model")
    saved_model = torch$load(sprintf("%s/%s_%d_%d/%sopt_train_saved_model.pth",
                                     dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm))

    print("Computing based on saved model...")
    #### IF NORMALIZE, NEED TO NORMALIZE Xs_aug$test...
    test_set = torch$Tensor(as.matrix( (Xs_aug$test - matrix(norm_means_x,nrow=nrow(Xs_aug$test),ncol=ncol(Xs_aug$test),byrow=T))/matrix(norm_sds_x,nrow=nrow(Xs_aug$test),ncol=ncol(Xs_aug$test),byrow=T) ))$cuda()
    test_set_out = invlink(link)(saved_model$NN_y(test_set))$detach()$cpu()$data$numpy()

    # Saving weights
    if(family=="Multinomial"){
      w0 = res$w0[-1] - res$w0[1] ## w0 should be nothing --> removed intercept to prevent multicollinearity
      w = res$w[-1,] - res$w[1,]    # reference is first class
      w_real = c(w[data_types_x=="real"])
      w_cat = c(w[data_types_x=="cat"])
    }else{
      w0 = res$w0 ## w0 should be nothing --> removed intercept to prevent multicollinearity
      w = res$w
      w_real = c(w[data_types_x=="real"])
      w_cat = c(w[data_types_x=="cat"])
    }

    if(family %in% c("Gaussian","Poisson")){
      # mu_y = as.matrix(Xs_aug$test) %*% t(res$w) + as.matrix(rep(res$w0, nrow(Xs_aug$test)),ncol=1)
      mu_y = test_set_out[,1]      # P(Y=k) with c=1 as reference, k=2, ..., K

      mu_y2 = res$all_params$y$mean  # average over the multiple samples of Xm --> Y'1
    } else if(family=="Multinomial"){
      # eta = as.matrix(Xs_aug$test) %*% t(res$w) + matrix(rep(res$w0, nrow(Xs_aug$test)),nrow=nrow(Xs_aug$test),ncol=nrow(res$w),byrow=T)
      # probs_y = t(apply(eta, 1, softmax))
      # mu_y = apply(probs_y,1,which.max)

      probs_y = test_set_out     # P(Y=k) with c=1 as reference, k=2, ..., K
      mu_y = apply(probs_y,1,which.max)
      mu_y2 = apply(res$all_params$y$probs, 1,which.max)   # using the p(y|x) posterior mode of test set --> why? could just predict

    }
    yhats = cbind(yhats,mu_y); colnames(yhats)[ncol(yhats)] = "dlglm"
    yhats2 = cbind(yhats2,mu_y2); colnames(yhats2)[ncol(yhats2)] = "dlglm"
    load(sprintf("%s/%s_%d_%d/%sopt_train.out", dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm))

    if(family=="Multinomial"){
      if(Cy==2){
        prhats$"dlglm" = probs_y[,-1]
        prhats2$"dlglm" = res$all_params$y$probs[,-1]   ## if only 2 classes, yield just one prob (pr(y=1))
        prhats_train$"dlglm" = res_train$all_params$y$probs[,-1]
      }else if(Cy>2){
        prhats$"dlglm" = probs_y
        prhats2$"dlglm" = res$all_params$y$probs   # if C classes, yield one prob per class
        prhats_train$"dlglm" = res_train$all_params$y$probs
      }
      # prhats$"dlglm_mode" = res$all_params$y$probs[,-1]
      if(Cy==2){
      }else if(Cy>2){
      }
    }

    if(res$train_params$n_hidden_layers_y == 0){
      tab = cbind(tab,c(w))
    }else{ tab = cbind(tab, 0)}

    colnames(tab)[ncol(tab)] = "dlglm"     ####### w: not right dim if HL's

    xhats$"dlglm" = res$xhat
    rm(test_set)
  }
  gc()
  if("idlglm" %in% methods){
    fname = sprintf("%s/Ignorable/res_dlglm_%s_%d_%d.RData",dir_name0,mechanism,miss_pct_features,pi*100)
    print(paste("idlglm Results file: ", fname))
    if(!file.exists(fname)){ break }
    load( fname )  # loads "X","Y","mask_x","mask_y","g"
    # should contain "res", which is a list that contains "results" and "fixed.params" from now on. First iteration only contains res (no list) results object
    ires = res$results
    iniws = ires$train_params$niws_z
    print("Loading saved model")

    saved_model = torch$load(sprintf("%s/Ignorable/%s_%d_%d/%sopt_train_saved_model.pth",dir_name0, mechanism, miss.params$miss_pct_features, pi*100, inorm))

    print("Computing based on saved model...")

    test_set = torch$Tensor(as.matrix( (Xs_aug$test - matrix(norm_means_x,nrow=nrow(Xs_aug$test),ncol=ncol(Xs_aug$test),byrow=T))/ matrix(norm_sds_x,nrow=nrow(Xs_aug$test),ncol=ncol(Xs_aug$test),byrow=T)))$cuda()
    test_set_out = invlink(link)(saved_model$NN_y(test_set))$detach()$cpu()$data$numpy()
    # Saving weights
    if(family=="Multinomial"){
      iw0 = ires$w0[-1] - ires$w0[1] ## w0 should be nothing --> removed intercept to prevent multicollinearity
      iw = ires$w[-1,] - ires$w[1,]    # reference is first class
      iw_real = c(iw[data_types_x=="real"])
      iw_cat = c(iw[data_types_x=="cat"])
    }else{
      iw0 = ires$w0 ## w0 should be nothing --> removed intercept to prevent multicollinearity
      iw = ires$w
      iw_real = c(iw[data_types_x=="real"])
      iw_cat = c(iw[data_types_x=="cat"])
    }

    if(family %in% c("Gaussian","Poisson")){
      # imu_y = as.matrix(Xs_aug$test) %*% t(ires$w) + as.matrix(rep(ires$w0, nrow(Xs_aug$test)),ncol=1)
      imu_y = test_set_out[,1]      # P(Y=k) with c=1 as reference, k=2, ..., K
      imu_y2 = ires$all_params$y$mean  # average over the multiple samples of Xm --> Y'1
    } else if(family=="Multinomial"){
      # ieta = as.matrix(Xs_aug$test) %*% t(ires$w) + matrix(rep(ires$w0, nrow(Xs_aug$test)),nrow=nrow(Xs_aug$test),ncol=nrow(ires$w),byrow=T)
      # iprobs_y = t(apply(ieta, 1, softmax))  # should this be softmax?
      # imu_y = apply(iprobs_y,1,which.max)
      iprobs_y = test_set_out     # P(Y=k) with c=1 as reference, k=2, ..., K
      imu_y = apply(iprobs_y,1,which.max)
      imu_y2 = apply(ires$all_params$y$probs, 1,which.max)   # using the p(y|x) posterior mode of test set --> why? could just predict
    }
    yhats = cbind(yhats,imu_y); colnames(yhats)[ncol(yhats)] = "idlglm"
    yhats2 = cbind(yhats2,imu_y2); colnames(yhats2)[ncol(yhats2)] = "idlglm"
    load(sprintf("%s/Ignorable/%s_%d_%d/%sopt_train.out", dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm))

    if(family=="Multinomial"){
      if(Cy==2){
        prhats$"idlglm" = iprobs_y[,-1]
        prhats2$"idlglm" = ires$all_params$y$probs[,-1]
        prhats_train$"idlglm" = res_train$all_params$y$probs[,-1]  # res_train will be idlglm now
      }else if(Cy>2){
        prhats$"idlglm" = iprobs_y
        prhats2$"idlglm" = ires$all_params$y$probs
        prhats_train$"idlglm" = res_train$all_params$y$probs
      }
      # prhats$"idlglm_mode" = ires$all_params$y$probs[,-1]
    }

    if(ires$train_params$n_hidden_layers_y == 0){
      tab = cbind(tab,c(iw))
    }else{ tab = cbind(tab, 0)}

    colnames(tab)[ncol(tab)] = "idlglm"

    xhats$"idlglm" = ires$xhat
    rm(test_set)
  }
  rm(Xs_aug)  # just need for dlglm
  gc()


  if("mice" %in% methods){
    fname_mice = sprintf("%s/res_mice_%s_%d_%d.RData",dir_name0,mechanism,miss_pct_features,pi*100)
    print(paste("mice Results file: ", fname_mice))
    if(!file.exists(fname_mice)){ break }

    load( fname_mice )
    print("Loaded mice results")
    xhat_mice = res_mice$xhat_mice; xyhat_mice = res_mice$xyhat_mice  # training x/yhats
    res_MICE = res_mice$res_MICE; res_MICE_test=res_mice$res_MICE_test
    rm(res_mice)
    if(family=="Gaussian"){
      fits_MICE = list()
      for(i in 1:res_MICE$m){
        if(i==1){
          xhat_mice_train = mice::complete(res_MICE,i)
        }else{ xhat_mice_train = xhat_mice_train + mice::complete(res_MICE,i) }
        fits_MICE[[i]] = glm(y ~ 0+., data=mice::complete(res_MICE,i))
        fits_MICE[[i]]$data = NULL; fits_MICE[[i]]$model=NULL
      }
      xhat_mice_train = xhat_mice_train/res_MICE$m
      fit_MICE = pool(fits_MICE)
    }else if(family=="Multinomial"){
      fits_MICE = list()
      for(i in 1:res_MICE$m){
        if(i==1){
          xhat_mice_train = mice::complete(res_MICE,i)
        }else{ xhat_mice_train = xhat_mice_train + mice::complete(res_MICE,i) }
        fits_MICE[[i]] = nnet::multinom(y ~ 0+., data=mice::complete(res_MICE,i))
        fits_MICE[[i]]$data = NULL; fits_MICE[[i]]$model=NULL
      }
      xhat_mice_train = xhat_mice_train/res_MICE$m
      fit_MICE = pool(fits_MICE)
    }
    xhat_mice_train = xhat_mice_train[,-ncol(xhat_mice_train)]

    dummy_fit_MICE = fits_MICE[[1]]; rm(fits_MICE)
    dummy_fit_MICE$coefficients = fit_MICE$pooled$estimate
    yhat_mice_pred = predict(dummy_fit_MICE,newdata = Xs$test)

    print("Completing test X")
    Xs_test_mice = mice::complete(res_MICE_test,1)
    for(j in 2:res_MICE_test$m){
      Xs_test_mice = Xs_test_mice + mice::complete(res_MICE_test,j)
    }
    Xs_test_mice = Xs_test_mice/res_MICE_test$m          # average of multiply-imputed test sets

    print("Predicting y")
    yhat_mice_pred2 = predict(dummy_fit_MICE,newdata = Xs_test_mice)

    w_mice = c(fit_MICE$pooled$estimate)

    yhats = cbind(yhats,as.numeric(yhat_mice_pred)); colnames(yhats)[ncol(yhats)] = "mice"
    yhats2 = cbind(yhats2,as.numeric(yhat_mice_pred2)); colnames(yhats2)[ncol(yhats2)] = "mice"

    if(family=="Multinomial"){
      prhat_mice_pred = predict(dummy_fit_MICE, newdata=Xs$test, type="probs")
      prhat_mice_pred2 = predict(dummy_fit_MICE, newdata=Xs_test_mice, type="probs")
      # if(Cy==2){
      #   prhats$"mice" = matrix(nrow=length(prhat_mice_pred),ncol=2); prhats$"mice"[,2] = prhat_mice_pred; prhats$"mice"[,1] = 1-prhats$"mice"[,2]
      #   prhats2$"mice" = matrix(nrow=length(prhat_mice_pred2),ncol=2); prhats2$"mice"[,2] = prhat_mice_pred2; prhats2$"mice"[,1] = 1-prhats2$"mice"[,2]
      # }else{
      #   prhats$"mice" = prhat_mice_pred
      #   prhats2$"mice" = prhat_mice_pred2
      # }
      colnames(prhat_mice_pred) = NULL;      colnames(prhat_mice_pred2) = NULL
      prhats$"mice" = prhat_mice_pred
      prhats2$"mice" = prhat_mice_pred2
      prhats_train$"mice" = predict(dummy_fit_MICE, newdata=xhat_mice_train, type="probs")

    }
    tab = cbind(tab,w_mice); colnames(tab)[ncol(tab)] = "mice"


    xhats$"mice" = xhat_mice
    rm(xhat_mice)
    rm(xyhat_mice)
  }
  gc()

  print("All glms fitted. Plotting diagnostics")

  # load simulated parameters file
  load( sprintf("%s/params_%s_%d_%d.RData", dir_name0, mechanism, miss_pct_features,pi*100) )
  if(grepl("SIM",dataset)){
    if(family=="Multinomial"){
      beta0s = sim.data$params$beta0s; betas = sim.data$params$betas[,-1] - sim.data$params$betas[,1]
      prs = split(data.frame(sim.data$params$prs[,-1]), g)   # first column: reference column.  # probabilities
      # prhats$"truth" = prs$test
    }else{ beta0s = sim.data$params$beta0s; betas = sim.data$params$betas }
    tab = cbind(betas,tab)

    if(family=="Gaussian"){
      true_mu_y = sim.data$params$beta0s + as.matrix(Xs$test) %*% as.matrix(sim.data$params$betas,ncol=1)
    }

    ## some dlglm-specific plots ##
    plots_dlglm = function(Ys_test, mu_y, family, dir_name, mechanism, miss_pct_features, pi,
                           betas, w_real, w_cat, data_types_x, data_types_x_0, Ignorable){

      if(family%in% c("Gaussian","Poisson")){
        if(Ignorable){ fname = sprintf("%s/%s_%d_%d/%spred_Y_idlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm)
        }else{ fname = sprintf("%s/%s_%d_%d/%spred_Y_dlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm) }

        png(fname)
        boxplot(as.numeric(unlist(Ys$test - as.numeric(mu_y))), main="Diff between test and pred Y",outline=F)  # prediction of Y (mu_y as)
        dev.off()
      } else if(family=="Multinomial"){
        tab = table(mu_y,unlist(Ys_test))
        p<-tableGrob(round(tab,3))

        library(mclust)
        ARI_val = mclust::adjustedRandIndex(mu_y,unlist(Ys_test))
        title <- textGrob(sprintf("Y, Pred(r), true(c), ARI=%f",round(ARI_val,3)),gp=gpar(fontsize=8))
        padding <- unit(5,"mm")

        p <- gtable_add_rows(
          p,
          heights = grobHeight(title) + padding,
          pos = 0)
        p <- gtable_add_grob(
          p,
          title,
          1, 1, 1, ncol(p))

        if(Ignorable){ fname = sprintf("%s/%s_%d_%d/%spred_classes_idlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm)
        }else{ fname = sprintf("%s/%s_%d_%d/%spred_classes_dlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm) }
        ggsave(p,file=fname, width=6, height=6, units="in")
      }

      fname = if(Ignorable){sprintf("%s/%s_%d_%d/%scoefs_real_idlglm.png",dir_name,mechanism, miss.params$miss_pct_features, pi*100, inorm)
      }else{sprintf("%s/%s_%d_%d/%scoefs_real_dlglm.png",dir_name,mechanism, miss.params$miss_pct_features, pi*100, inorm)}
      png(fname,res = 300,width = 4, height = 4, units = 'in')
      ymin = min(c(betas[data_types_x_0=="real"], w_real))
      ymax = max(c(betas[data_types_x_0=="real"], w_real))
      plot(c(betas[data_types_x_0=="real"]), main="True (black) vs fitted (red) real coefs", xlab="covariate", ylab="coefficient", ylim=c(ymin,ymax)); points(c(w_real),col="red", cex=0.5)
      dev.off()
      if(sum(data_types_x=="cat")>0){
        fname = if(Ignorable){sprintf("%s/%s_%d_%d/%scoefs_cat_idlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm)
        }else{sprintf("%s/%s_%d_%d/%scoefs_cat_dlglm.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm)}
        png(fname,res = 300,width = 4, height = 4, units = 'in')
        ymin = min(c(betas[data_types_x_0=="cat"],sapply(split(w_cat, rep(1:length(Cs), Cs)), function(x){mean((x - c(0,x[-length(x)]))[-1])})))-0.005
        ymax = max(c(betas[data_types_x_0=="cat"],sapply(split(w_cat, rep(1:length(Cs), Cs)), function(x){mean((x - c(0,x[-length(x)]))[-1])})))+0.005
        plot(c(betas[data_types_x_0=="cat"]), main="True (black) vs fitted (red) cat coefs", xlab="covariate", ylab="coefficient", ylim=c(ymin,ymax))
        points(sapply(split(w_cat, rep(1:length(Cs), Cs)), function(x){mean((x - c(0,x[-length(x)]))[-1])}),col="red", cex=0.5)
        dev.off()
      }
    }

    if("dlglm" %in% methods){ plots_dlglm(Ys$test, mu_y, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                          betas, w_real, w_cat, data_types_x, data_types_x_0, F) }
    if("idlglm" %in% methods){ plots_dlglm(Ys$test, imu_y, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                           betas, iw_real, iw_cat, data_types_x, data_types_x_0, T) }
    ###############################

    plots_others = function(Ys_test, yhat, family, dir_name, mechanism, miss_pct_features, pi,
                            betas, w, data_types_x_0, method){
      png(sprintf("%s/%s_%d_%d/%scoefs_real_%s.png",dir_name,mechanism, miss.params$miss_pct_features, pi*100, method, inorm),
          res = 300,width = 4, height = 4, units = 'in')
      ymin = min(c(betas[data_types_x_0=="real"], w))
      ymax = max(c(betas[data_types_x_0=="real"], w))
      plot(c(betas[data_types_x_0=="real"]), main="True (black) vs fitted (red) real coefs", xlab="covariate", ylab="coefficient", ylim=c(ymin,ymax)); points(c(w),col="red", cex=0.5)
      dev.off()

      if(family%in% c("Gaussian","Poisson")){
        fname = sprintf("%s/%s_%d_%d/%spred_Y_%s.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100,method, inorm)
        png(fname)
        boxplot(as.numeric(unlist(Ys_test - as.numeric(yhat))), main="Diff between test and pred Y",outline=F)  # prediction of Y (mu_y as)
        dev.off()
      } else if(family=="Multinomial"){
        tab = table(yhat,unlist(Ys_test))
        p<-tableGrob(round(tab,3))

        library(mclust)
        ARI_val = adjustedRandIndex(yhat,unlist(Ys_test))
        title <- textGrob(sprintf("Y, Pred(r), true(c), ARI=%f",round(ARI_val,3)),gp=gpar(fontsize=8))
        padding <- unit(5,"mm")

        p <- gtable_add_rows(
          p,
          heights = grobHeight(title) + padding,
          pos = 0)
        p <- gtable_add_grob(
          p,
          title,
          1, 1, 1, ncol(p))

        fname = sprintf("%s/%s_%d_%d/%spred_classes_%s.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100,method, inorm)
        ggsave(p,file=fname, width=6, height=6, units="in")
      }
    }
    if("zero" %in% methods){ plots_others(Ys$test, yhat_zero_pred, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                          betas, w_zero, data_types_x_0, "zero") }
    if("mean" %in% methods){ plots_others(Ys$test, yhat_mean_pred, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                          betas, w_mean, data_types_x_0, "mean") }
    if("mice" %in% methods){ plots_others(Ys$test, yhat_mice_pred, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                          betas, w_mice, data_types_x_0, "mice") }
  }


  # comparing all
  L1s_y = abs( matrix(unlist(Ys$test),nrow=nrow(Ys$test),ncol=ncol(yhats)) - yhats )
  L1s_y2 = abs( matrix(unlist(Ys$test),nrow=nrow(Ys$test),ncol=ncol(yhats2)) - yhats2 )
  if(family%in% c("Gaussian","Poisson")){
    png(filename=sprintf("%s/%s_%d_%d/%spredY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),res = 300,width = 6, height = 6, units = 'in')
    boxplot(L1s_y,names=colnames(L1s_y), outline=F,
            main="Abs Diff: test Y and predicted Y")
    # boxplot(as.numeric(unlist(abs(Ys$test - mu_y))),
    #         as.numeric(unlist(abs(Ys$test - as.numeric(yhat_zero_pred)))),
    #         names=c("dlglm","zero"), outline=F,
    #         main="Absolute Difference between test Y and predicted Y")
    dev.off()

    png(filename=sprintf("%s/%s_%d_%d/%spredY_missTestX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),res = 300,width = 6, height = 6, units = 'in')
    boxplot(L1s_y2,names=colnames(L1s_y2), outline=F,
            main="Abs Diff: test Y and pred Y (miss testX)")
    # boxplot(as.numeric(unlist(abs(Ys$test - mu_y))),
    #         as.numeric(unlist(abs(Ys$test - as.numeric(yhat_zero_pred)))),
    #         names=c("dlglm","zero"), outline=F,
    #         main="Absolute Difference between test Y and predicted Y")
    dev.off()

  }

  if(family=="Multinomial" & grepl("SIM",dataset)){
    p = tableGrob(round(t(unlist(lapply(prhats,function(x){mean(abs(x - c(prs$test)[[1]]))}))), 3))
    title <- textGrob("Mean Diff Between True vs Pred Probs_y",gp=gpar(fontsize=9))
    padding <- unit(5,"mm")

    p <- gtable_add_rows(
      p,
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p,
      title,
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/%smean_L1_probY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm), width=6, height=4, units="in")

    prs_L1 = lapply(prhats, function(x){abs(x - c(prs$test)[[1]])})
    png(filename=sprintf("%s/%s_%d_%d/%sL1s_probY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),res = 300,width = 10, height = 10, units = 'in')
    boxplot(prs_L1, names=names(prs_L1),
            outline=F, main="Abs deviation between true and pred Pr(Y)")
    dev.off()

    p = tableGrob(round(t(unlist(lapply(prhats2,function(x){mean(abs(x - c(prs$test)[[1]]))}))), 3))
    title <- textGrob("Mean Diff: True vs Pred Probs_y (missing testX)",gp=gpar(fontsize=9))
    padding <- unit(5,"mm")

    p <- gtable_add_rows(
      p,
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p,
      title,
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/%smean_L1_probY_missTestX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm), width=6, height=4, units="in")

    prs_L1 = lapply(prhats2, function(x){abs(x - c(prs$test)[[1]])})
    png(filename=sprintf("%s/%s_%d_%d/%sL1s_probY_missTestX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),res = 300,width = 10, height = 10, units = 'in')
    boxplot(prs_L1, names=names(prs_L1),
            outline=F, main="Abs deviation: true and pred Pr(Y) (missing testX)")
    dev.off()
  }


  ################### TOGETHER PLOTS
  if(grepl("SIM",dataset)){
    p<-tableGrob(round(tab,3))
    title <- textGrob("Wts from dlglm/idlglm, other coefs imp + glm (no int)",gp=gpar(fontsize=9))
    padding <- unit(5,"mm")

    p <- gtable_add_rows(
      p,
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p,
      title,
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/%scoefs_tab.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm), width=6, height=25, units="in")


    ## Compute RB, PB, AW, CR (no CR/AW b/c no 95% CI for dlglm coef est's)
    RB = tab[,-1] - tab[,1]
    PB = 100*abs(tab[,-1] - tab[,1])/max(abs(tab[,1]), 1e-10)  # PB will blow up to a very large number if true coef est is 0
    SE = sqrt((tab[,-1] - tab[,1])^2)   # sqrt/mean across sim index to get RMSE

    df = cbind(RB, PB, SE)
    df = rbind(df, c( colMeans(RB), colMeans(PB), colMeans(SE) ))
    rownames(df) = c(1:length(c(betas)), "Averaged")

    colnames(tab)[-1]
    colnames(df) = paste(rep(c("RB","PB","SE"),each=ncol(tab)-1),
                         colnames(tab)[-1],sep="_")

    p<-tableGrob(round(df,3))
    title <- textGrob("Raw and Percent Bias (RB/PB), and Squared error (SE) of dlglm (dl), mean (me), zero (ze), and mice (mi)",gp=gpar(fontsize=9))
    padding <- unit(5,"mm")

    p <- gtable_add_rows(
      p,
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p,
      title,
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/%scoefs_tab2.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm), width=25, height=floor(P/2), units="in", limitsize = FALSE)
  }

  ### CANT DO CR WITH DLGLM ###
  # summ_dlglm = cbind(tab[,2], NA,NA,NA)  # no CI for dlglm
  # summ_mean = cbind(summary(fit_mean)$coefficients, summary(fit_mean)$standard.errors, confint(fit_mean)); colnames(summ_mean) = c("estimate", "std.error", "2.5 %", "97.5 %")
  # summ_zero = cbind(summary(fit_zero)$coefficients, summary(fit_zero)$standard.errors, confint(fit_zero)); colnames(summ_zero) = c("estimate", "std.error", "2.5 %", "97.5 %")
  # summ_mice = summary(fit_MICE, "all", conf.int = TRUE)[, c("estimate", "std.error", "2.5 %", "97.5 %")]
  # coverage_width = function(summ, truth){
  #   # summ: contains 4 columns: estimate, std.error, 2.5 %, and 97.5 %
  #   # truth is just the true values
  #    coverage = (summ[,"2.5 %"] < truth & summ[,"97.5 %"] > truth)^2
  #    width = summ[,"97.5 %"] - summ[,"2.5 %"]
  #
  #    return(list(coverage=coverage, width=width))
  # }
  # CR = cbind( coverage_width(summ_mean, tab[,1])$coverage,
  #              coverage_width(summ_zero, tab[,1])$coverage,
  #              coverage_width(summ_mice, tab[,1])$coverage)
  # AW = cbind( coverage_width(summ_mean, tab[,1])$width,
  #              coverage_width(summ_zero, tab[,1])$width,
  #              coverage_width(summ_mice, tab[,1])$width)
  #
  # df = cbind(CR, AW)
  # df = rbind(df, c( colMeans(CR), colMeans(AW) ))
  # rownames(df) = c(1:length(c(betas)), "Averaged")
  # colnames(df) = c("CR_me", "CR_ze", "CR_mi",
  #                  "AW_me", "AW_ze", "AW_mi")
  #
  # p<-tableGrob(round(df,3))
  # title <- textGrob("Coverage (CR) and Average width (AW) of CI's of estimates from mean (me), zero (ze), and mice (mi)",gp=gpar(fontsize=9))
  # padding <- unit(5,"mm")
  #
  # p <- gtable_add_rows(
  #   p,
  #   heights = grobHeight(title) + padding,
  #   pos = 0)
  # p <- gtable_add_grob(
  #   p,
  #   title,
  #   1, 1, 1, ncol(p))
  # ggsave(p,file=sprintf("%s/%s_%d_%d/coefs_tab3.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=12, height=15, units="in")



  rel_diffs = L1s_y/max(abs(Ys$test),0.001)
  pct_diffs = colMeans(rel_diffs)

  png(filename=sprintf("%s/%s_%d_%d/%spredY_rel.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),res = 300,width = 12, height = 12, units = 'in')
  boxplot(rel_diffs, outline=F,
          names=colnames(rel_diffs),
          main="Relative Difference between test Y and predicted Y")
  dev.off()

  p<-tableGrob(t(round(pct_diffs,3)))
  title <- textGrob("Rel diff between true and pred Ys",gp=gpar(fontsize=8.5))
  padding <- unit(5,"mm")
  p <- gtable_add_rows(
    p,
    heights = grobHeight(title) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p,
    title,
    1, 1, 1, ncol(p))
  ggsave(p,file=sprintf("%s/%s_%d_%d/%smean_pctdiff_predY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm), width=6, height=4, units="in")

  ###########################################
  ###### look at imputation in missing ######
  ###########################################
  print("Imputation")
  diffs = lapply(xhats, function(x) abs(x - Xs$test))
  L1s = lapply(diffs, function(x) x[Rxs$test==0])
  pcts = lapply(diffs, function(x) (x/max(abs(Xs$test), 0.001))[Rxs$test==0])
  # Mean/zero imputation results:

  # compare both imputation results
  if(grepl("x",case)){
    # for(c in miss_cols){
    #   ids = Rxs$test[,c]==0
    #   diffs_df = lapply(diffs,function(x) x[ids,c])
    #   png(filename=sprintf("%s/%s_%d_%d/%simputedX_col%d.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm,c),res = 300,width = 10, height = 10, units = 'in')
    #   boxplot(diffs_df, names=names(diffs_df),
    #           outline=F, main=sprintf("Abs dev between true and imputed X in col%d",c))
    #   dev.off()
    # }
    png(filename=sprintf("%s/%s_%d_%d/%simputedX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),res = 300,width = 10, height = 10, units = 'in')
    boxplot(L1s, names=names(L1s),
            outline=F, main="Absolute deviation between true and imputed X")
    dev.off()
    png(filename=sprintf("%s/%s_%d_%d/%simputedpctX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm),
        res = 300,width = 10, height = 10, units = 'in')
    boxplot(pcts, names=names(pcts),
            outline=F, main="Relative difference between true and imputed X")
    dev.off()


    # tiff(filename=sprintf("%s/%s_%d_%d/scatter_imputedX_refzero.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
    # plot(x=L1s_x-L1s_zero_x,y=L1s_mean_x-L1s_zero_x, main="Err dlglm vs mean imputed X, wrt zero-imputed X")
    # dev.off()
    #
    # tiff(filename=sprintf("%s/%s_%d_%d/scatter_imputedX_refzero_mice.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
    # plot(x=L1s_mice_x-L1s_zero_x,y=L1s_mean_x-L1s_zero_x, main="Err mice vs mean imputed X, wrt zero-imputed X")
    # dev.off()

    tab3 = t(unlist(lapply(L1s,mean)))
    p<-tableGrob(round(tab3,3))
    title <- textGrob("Mean abs diff: true - imputed Xs",gp=gpar(fontsize=8.5))
    p <- gtable_add_rows(
      p,
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p,
      title,
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/%smean_L1s_X.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100, inorm), width=6, height=4, units="in")
  }
  # if(grepl("y",case)){
  #   tiff(filename=sprintf("%s/%s_%d_%d/imputedY.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
  #   boxplot(L1s_y,L1s_mean_y,L1s_zero_y,
  #           names=c("dlglm","mean","zero"),
  #           outline=F, main="Absolute deviation between true and imputed Y")
  #   dev.off()
  #   tiff(filename=sprintf("%s/%s_%d_%d/imputedpctY.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
  #   boxplot(pcts_y,pcts_mean_y,pcts_zero_y,
  #           names=c("dlglm","mean","zero"),
  #           outline=F, main="Relative difference between true and imputed Y")
  #   dev.off()
  #
  #   tiff(filename=sprintf("%s/%s_%d_%d/imputedY_refzero.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
  #   boxplot(L1s_y-L1s_zero_y,
  #           names=c("dlglm"),
  #           outline=F, main="Err imputed Y - Err zero-imputed Y")
  #   abline(h=L1s_mean_y[1]-L1s_zero_y[1], col = "Red")
  #   dev.off()
  # }

  ## TRYING: glm on dlglm resulting xhat (test data)
  # d_dlglm = cbind( xhat, Ys$test )
  # fit_dlglm = glm(Y ~ 0 + . , data=d_dlglm)

  # # mean - dlglm pairs plot
  # library(GGally)
  # my_fn = function(data, mapping, ...){
  #   p <- ggplot(data=data,mapping=mapping) + geom_point() + geom_abline(slope=1, intercept=0, colour="red")
  #   p
  # }
  # # for(c in miss_cols){
  #   # mask_ids = Rxs$test[,c]==0
  #   p = ggpairs(data.frame(Ys$test,
  #                            "dlglm"=as.numeric(mu_y),
  #                            "mean"=yhat_mean_pred,
  #                            "zero"=yhat_zero_pred,
  #                             "mice"=yhat_mice_pred), lower=list(continuous=my_fn),
  #               title="Plots of true vs predicted values of Y")
  #   ggsave(sprintf("%s/%s_%d_%d/pairs.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),p,height=6, width=6, units="in")
  #   df = data.frame("dlglm"=Ys$test-as.numeric(mu_y),
  #                   "mean"=Ys$test-as.numeric(yhat_mean_pred),
  #                   "zero"=Ys$test-as.numeric(yhat_zero_pred),
  #                   "mice"=Ys$test-as.numeric(yhat_mice_pred))
  #   names(df) = c("dlglm","mean","zero","mice")
  #   p = ggpairs(df,lower=list(continuous=my_fn),
  #               title="Plots of diffs between true vs pred values of Y")
  #   ggsave(sprintf("%s/%s_%d_%d/pairs_diff.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),p,height=6, width=6, units="in")
  #   df=data.frame("dlglm"=abs(Ys$test-as.numeric(mu_y)),
  #                 "mean"=abs(Ys$test-as.numeric(yhat_mean_pred)),
  #                 "zero"=abs(Ys$test-as.numeric(yhat_zero_pred)),
  #                 "mice"=abs(Ys$test-as.numeric(yhat_mice_pred)))
  #   names(df) = c("dlglm","mean","zero","mice")
  #   p = ggpairs(df,lower=list(continuous=my_fn),
  #               title="Plots of abs diffs between true vs pred values of Y")
  #   ggsave(sprintf("%s/%s_%d_%d/pairs_absdiff.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),p,height=6, width=6, units="in")
  # # }

  # ### IMPUTATION: average L1 distance between xhat and truth for just missing cols
  # L1s
  #
  # ### INFERENCE: (RB, PB, and SE) for all methods, one rep
  # df
  #
  # ### PREDICTION: mean diff in prob simulated vs predicted (complete testX p, and imputed testX p2)
  # p = t(unlist(lapply(prhats,function(x){mean(abs(x - c(prs$test)[[1]]))})))
  # p2 = t(unlist(lapply(prhats2,function(x){mean(abs(x - c(prs$test)[[1]]))})))
  print("Prediction")
  yhats = asplit(yhats,2)
  yhats2 = asplit(yhats2,2)

  cutoffs = seq(0.01,1,0.01)

  ### use prhats_train list to tune cutoff: mice, mean, dlglm, idlglm. probs of training set.
  ### NEED TO TUNE CUTOFFS FOR: PPV, TPR, FPR, F1, kappa, and mcc
  ## measures::PPV(truth, response, positive, probabilities = NULL)
  library(yardstick)

  if(family=="Multinomial"){
    all_classes = unique(Ys$test$Y)
    if(Cy==2){
      yhats_cut = list(); yhats_cut2 = list()
      # for(i in 1:length(prhats)){
      for(m in 1:length(methods)){
        cut_metrics = matrix(nrow=length(cutoffs),ncol=6)
        colnames(cut_metrics) = c("PPV","TPR","FPR","F1","kappa","MCC"); rownames(cut_metrics) = cutoffs
        for(c in 1:length(cutoffs)){
          yhat = (prhats_train[which(names(prhats_train)==methods[m])][[1]]>=cutoffs[c])^2*max(Ys$train$Y) + (prhats_train[which(names(prhats_train)==methods[m])][[1]]<cutoffs[c])^2*min(Ys$train$Y)
          #### use kappa as main metric
          ppv=measures::PPV(Ys$train$Y, yhat, max(Ys$train$Y))
          tpr=measures::TPR(Ys$train$Y, yhat, max(Ys$train$Y))
          fpr=measures::FPR(Ys$train$Y, yhat, min(Ys$train$Y), max(Ys$train$Y))
          f1=measures::F1(Ys$train$Y, yhat, max(Ys$train$Y))
          kap=measures::KAPPA(Ys$train$Y, yhat)
          mcc=yardstick::mcc_vec(truth=factor(c(Ys$train$Y),levels=levels(factor(Ys$train$Y))), estimate=factor(yhat,levels=levels(factor(Ys$train$Y))))
          cut_metrics[c,] = c(ppv,tpr,fpr,f1,kap,mcc)
        }
        opt_cutoff = as.numeric(names(which.max(cut_metrics[,"kappa"])))
        yhats_cut[[m]] = (prhats[which(names(prhats)==methods[m])][[1]]>opt_cutoff)^2*max(Ys$train$Y) + (prhats[which(names(prhats)==methods[m])][[1]]<=opt_cutoff)^2*min(Ys$train$Y)    # apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
        yhats_cut2[[m]] = (prhats2[which(names(prhats2)==methods[m])][[1]]>opt_cutoff)^2*max(Ys$train$Y) + (prhats2[which(names(prhats2)==methods[m])][[1]]<=opt_cutoff)^2*min(Ys$train$Y)    # apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
        names(yhats_cut)[m] = methods[m]; names(yhats_cut2)[m] = methods[m]
      }
      AUC = lapply(prhats,function(x){as.numeric(pROC::auc(c(Ys$test$Y), x))})
      AUC2 = lapply(prhats2,function(x){as.numeric(pROC::auc(c(Ys$test$Y), x))})   ### CAT ONLY

      br1 = lapply(prhats, function(x){measures::Brier(x,c(Ys$test$Y), min(Ys$test$Y), max(Ys$test$Y))})
      br2 = lapply(prhats2, function(x){measures::Brier(x,c(Ys$test$Y), min(Ys$test$Y), max(Ys$test$Y))})  ### BINARY ONLY

      # TPR1 = lapply(yhats, function(x){if(all(Ys$test$Y != x)){0} else{measures::TPR(c(Ys$test$Y), x, max(Ys$test$Y))}})
      # TPR2 = lapply(yhats2, function(x){if(all(Ys$test$Y != x)){0} else{measures::TPR(c(Ys$test$Y), x, max(Ys$test$Y))}})
      # FPR1 = lapply(yhats, function(x){measures::FPR(c(Ys$test$Y), x, min(Ys$test$Y), max(Ys$test$Y))})
      # FPR2 = lapply(yhats2, function(x){measures::FPR(c(Ys$test$Y), x, min(Ys$test$Y), max(Ys$test$Y))})
      # F11 = lapply(yhats, function(x){measures::F1(c(Ys$test$Y), x, max(Ys$test$Y))})
      # F12 = lapply(yhats2, function(x){measures::F1(c(Ys$test$Y), x, max(Ys$test$Y))})
      # PPV1 = lapply(yhats, function(x){measures::PPV(c(Ys$test$Y), x, max(Ys$test$Y))})
      # PPV2 = lapply(yhats2, function(x){measures::PPV(c(Ys$test$Y), x, max(Ys$test$Y))})

      TPR1 = lapply(yhats_cut, function(x){if(all(Ys$test$Y != x)){0} else{measures::TPR(c(Ys$test$Y), x, max(Ys$test$Y))}})
      TPR2 = lapply(yhats_cut2, function(x){if(all(Ys$test$Y != x)){0} else{measures::TPR(c(Ys$test$Y), x, max(Ys$test$Y))}})
      FPR1 = lapply(yhats_cut, function(x){measures::FPR(c(Ys$test$Y), x, min(Ys$test$Y), max(Ys$test$Y))})
      FPR2 = lapply(yhats_cut2, function(x){measures::FPR(c(Ys$test$Y), x, min(Ys$test$Y), max(Ys$test$Y))})
      F11 = lapply(yhats_cut, function(x){measures::F1(c(Ys$test$Y), x, max(Ys$test$Y))})
      F12 = lapply(yhats_cut2, function(x){measures::F1(c(Ys$test$Y), x, max(Ys$test$Y))})
      PPV1 = lapply(yhats_cut, function(x){measures::PPV(c(Ys$test$Y), x, max(Ys$test$Y))})
      PPV2 = lapply(yhats_cut2, function(x){measures::PPV(c(Ys$test$Y), x, max(Ys$test$Y))})
      all_AUC = AUC; all_AUC2 = AUC2; all_br1 = br1; all_br2 = br2; all_TPR1=TPR1; all_TPR2=TPR2
      all_FPR1=FPR1; all_FPR2=FPR2; all_F11=F11; all_F12=F12; all_PPV1 = PPV1; all_PPV2 = PPV2
    } else{
      #### if more than one class, compute AUC and Brier scores on class 1 vs rest, then class 2 vs rest, ... . Then average?
      all_AUC = list(); all_AUC2 = list(); all_br1=list(); all_br2=list(); all_TPR1=list(); all_TPR2=list()
      all_FPR1=list(); all_FPR2=list(); all_F11=list(); all_F12=list(); all_PPV1=list(); all_PPV2=list()

      # for(i in 1:length(prhats)){
      #   for(c in 1:length(cutoffs)){
      #     yhat = apply(prhats_train[[i]],1, function(x){(x[2]>cutoffs[c])^2})
      #   }
      #   yhats_cut[[i]] = apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
      #   names(yhats_cut)[i] = names(prhats)[i]
      # }

      yhats_cut = list(); yhats_cut2 = list()
      # for(i in 1:length(prhats)){
      # for(m in 1:length(methods)){
      #   cut_metrics = matrix(nrow=length(cutoffs),ncol=6)
      #   colnames(cut_metrics) = c("PPV","TPR","FPR","F1","kappa","MCC"); rownames(cut_metrics) = cutoffs
      #   for(c in 1:length(cutoffs)){
      #     yhat = (prhats_train[which(names(prhats_train)==methods[m])][[1]]>=cutoffs[c])^2*max(Ys$train$Y) + (prhats_train[which(names(prhats_train)==methods[m])][[1]]<cutoffs[c])^2*min(Ys$train$Y)
      #     #### use kappa as main metric
      #     ppv=measures::PPV(Ys$train$Y, yhat, max(Ys$train$Y))
      #     tpr=measures::TPR(Ys$train$Y, yhat, max(Ys$train$Y))
      #     fpr=measures::FPR(Ys$train$Y, yhat, min(Ys$train$Y), max(Ys$train$Y))
      #     f1=measures::F1(Ys$train$Y, yhat, max(Ys$train$Y))
      #     kap=measures::KAPPA(Ys$train$Y, yhat)
      #     mcc=yardstick::mcc_vec(truth=factor(c(Ys$train$Y),levels=levels(factor(Ys$train$Y))), estimate=factor(yhat,levels=levels(factor(Ys$train$Y))))
      #     cut_metrics[c,] = c(ppv,tpr,fpr,f1,kap,mcc)
      #   }
      #   opt_cutoff = as.numeric(names(which.max(cut_metrics[,"kappa"])))
      #   yhats_cut[[m]] = (prhats[which(names(prhats)==methods[m])][[1]]>opt_cutoff)^2*max(Ys$train$Y) + (prhats[which(names(prhats)==methods[m])][[1]]<=opt_cutoff)^2*min(Ys$train$Y)    # apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
      #   yhats_cut2[[m]] = (prhats2[which(names(prhats2)==methods[m])][[1]]>opt_cutoff)^2*max(Ys$train$Y) + (prhats2[which(names(prhats2)==methods[m])][[1]]<=opt_cutoff)^2*min(Ys$train$Y)    # apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
      #   names(yhats_cut)[m] = methods[m]; names(yhats_cut2)[m] = methods[m]
      # }

      for(m in 1:length(methods)){
        all_AUC[[m]] = rep(NA, Cy); names(all_AUC)[m]=methods[m]; names(all_AUC[[m]]) = all_classes
        all_AUC2[[m]] = rep(NA, Cy); names(all_AUC2)[m]=methods[m]; names(all_AUC2[[m]]) = all_classes
        all_br1[[m]] = rep(NA, Cy); names(all_br1)[m]=methods[m]; names(all_br1[[m]]) = all_classes
        all_br2[[m]] = rep(NA, Cy); names(all_br2)[m]=methods[m]; names(all_br2[[m]]) = all_classes
        all_TPR1[[m]] = rep(NA, Cy); names(all_TPR1)[m]=methods[m]; names(all_TPR1[[m]]) = all_classes
        all_TPR2[[m]] = rep(NA, Cy); names(all_TPR2)[m]=methods[m]; names(all_TPR2[[m]]) = all_classes
        all_FPR1[[m]] = rep(NA, Cy); names(all_FPR1)[m]=methods[m]; names(all_FPR1[[m]]) = all_classes
        all_FPR2[[m]] = rep(NA, Cy); names(all_FPR2)[m]=methods[m]; names(all_FPR2[[m]]) = all_classes
        all_F11[[m]] = rep(NA, Cy); names(all_F11)[m]=methods[m]; names(all_F11[[m]]) = all_classes
        all_F12[[m]] = rep(NA, Cy); names(all_F12)[m]=methods[m]; names(all_F12[[m]]) = all_classes
        all_PPV1[[m]] = rep(NA, Cy); names(all_PPV1)[m]=methods[m]; names(all_PPV1[[m]]) = all_classes
        all_PPV2[[m]] = rep(NA, Cy); names(all_PPV2)[m]=methods[m]; names(all_PPV2[[m]]) = all_classes

        for(c in 1:Cy){
          print(c)
          Yc = c(Ys$test$Y); Yc[Yc!=all_classes[c]] = -999   # negative class dummy set as -999

          ###################################################
          Yc_train = c(Ys$train$Y); Yc_train[Yc_train!=all_classes[c]] = -999
          cut_metrics = matrix(nrow=length(cutoffs),ncol=6)
          colnames(cut_metrics) = c("PPV","TPR","FPR","F1","kappa","MCC"); rownames(cut_metrics) = cutoffs
          prhats_train_cut = prhats_train[which(names(prhats_train)==methods[m])][[1]][,all_classes[c]]
          prhats_cut = prhats[which(names(prhats)==methods[m])][[1]][,all_classes[c]]
          prhats_cut2 = prhats2[which(names(prhats2)==methods[m])][[1]][,all_classes[c]]

          # prhats_train_cut = prhats_train[which(names(prhats_train)==methods[m])][[1]][,c]
          # prhats_cut = prhats[which(names(prhats)==methods[m])][[1]][,c]
          # prhats_cut2 = prhats2[which(names(prhats2)==methods[m])][[1]][,c]

          for(co in 1:length(cutoffs)){
            yhat = all_classes[c]*(prhats_train_cut>=cutoffs[co])^2 +
              (-999)*(prhats_train_cut<cutoffs[co])^2
            #### use kappa as main metric
            ppv=measures::PPV(Yc_train, yhat, all_classes[c])
            tpr=measures::TPR(Yc_train, yhat, all_classes[c])
            fpr=measures::FPR(Yc_train, yhat, -999, all_classes[c])
            f1=measures::F1(Yc_train, yhat, all_classes[c])
            kap=measures::KAPPA(Yc_train, yhat)
            mcc=yardstick::mcc_vec(truth=factor(c(Yc_train),levels=levels(factor(Yc_train))), estimate=factor(yhat,levels=levels(factor(Yc_train))))
            cut_metrics[co,] = c(ppv,tpr,fpr,f1,kap,mcc)
          }

          opt_cutoff = as.numeric(names(which.max(cut_metrics[,"kappa"])))
          yhats_cut[[m]] = all_classes[c]*(prhats_cut>=opt_cutoff)^2 + (-999)*(prhats_cut<opt_cutoff)^2    # apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
          yhats_cut2[[m]] = all_classes[c]*(prhats_cut2>=opt_cutoff)^2 + (-999)*(prhats_cut2<opt_cutoff)^2    # apply(prhats,1, function(x){(x[2]>opt_cutoff)^2})
          names(yhats_cut)[m] = methods[m]; names(yhats_cut2)[m] = methods[m]

          ########################################

          all_AUC[[m]][c] = as.numeric(pROC::auc(Yc, prhats[names(prhats)==methods[m]][[1]][,c], levels=c(-999, all_classes[c]) ))
          all_AUC2[[m]][c] = as.numeric(pROC::auc(Yc, prhats2[names(prhats2)==methods[m]][[1]][,c], levels=c(-999, all_classes[c]) ))
          all_br1[[m]][c] = measures::Brier(prhats[names(prhats)==methods[m]][[1]][,c], Yc, -999, all_classes[c])
          all_br2[[m]][c] = measures::Brier(prhats2[names(prhats2)==methods[m]][[1]][,c], Yc, -999, all_classes[c])
          # if(all(yhats[names(yhats)==methods[m]][[1]] != Yc)){  # no true positives
          #   all_TPR1[[m]][c] = 0
          # }else{
          #   all_TPR1[[m]][c] = measures::TPR(yhats[names(yhats)==methods[m]][[1]], Yc, all_classes[c])
          # }
          # if(all(yhats2[names(yhats2)==methods[m]][[1]] != Yc)){
          #   all_TPR2[[m]][c] = 0
          # }else{
          #   all_TPR2[[m]][c] = measures::TPR(yhats2[names(yhats2)==methods[m]][[1]], Yc, all_classes[c])
          # }
          # yhats0 = yhats[names(yhats)==methods[m]][[1]]; yhats0[yhats0!=all_classes[c]]=-999
          # yhats20 = yhats2[names(yhats2)==methods[m]][[1]]; yhats20[yhats20!=all_classes[c]]=-999
          # all_FPR1[[m]][c] = measures::FPR(yhats0, Yc, -999, all_classes[c])
          # all_FPR2[[m]][c] = measures::FPR(yhats20, Yc, -999, all_classes[c])
          # all_F11[[m]][c] = measures::F1(yhats[names(yhats)==methods[m]][[1]], Yc, all_classes[c])
          # all_F12[[m]][c] = measures::F1(yhats2[names(yhats2)==methods[m]][[1]], Yc, all_classes[c])
          # all_PPV1[[m]][c] = measures::PPV(yhats[names(yhats)==methods[m]][[1]], Yc, all_classes[c])
          # all_PPV2[[m]][c] = measures::PPV(yhats2[names(yhats2)==methods[m]][[1]], Yc, all_classes[c])


          if(all(yhats_cut[names(yhats_cut)==methods[m]][[1]] != Yc)){  # no true positives
            all_TPR1[[m]][c] = 0
          }else{
            all_TPR1[[m]][c] = measures::TPR(yhats_cut[names(yhats_cut)==methods[m]][[1]], Yc, all_classes[c])
          }
          if(all(yhats_cut2[names(yhats_cut2)==methods[m]][[1]] != Yc)){
            all_TPR2[[m]][c] = 0
          }else{
            all_TPR2[[m]][c] = measures::TPR(yhats_cut2[names(yhats_cut2)==methods[m]][[1]], Yc, all_classes[c])
          }
          yhats0 = yhats_cut[names(yhats_cut)==methods[m]][[1]]; yhats0[yhats0!=all_classes[c]]=-999
          yhats20 = yhats_cut2[names(yhats_cut2)==methods[m]][[1]]; yhats20[yhats20!=all_classes[c]]=-999
          all_FPR1[[m]][c] = measures::FPR(yhats0, Yc, -999, all_classes[c])
          all_FPR2[[m]][c] = measures::FPR(yhats20, Yc, -999, all_classes[c])
          all_F11[[m]][c] = measures::F1(yhats_cut[names(yhats_cut)==methods[m]][[1]], Yc, all_classes[c])
          all_F12[[m]][c] = measures::F1(yhats_cut2[names(yhats_cut2)==methods[m]][[1]], Yc, all_classes[c])
          all_PPV1[[m]][c] = measures::PPV(yhats_cut[names(yhats_cut)==methods[m]][[1]], Yc, all_classes[c])
          all_PPV2[[m]][c] = measures::PPV(yhats_cut2[names(yhats_cut2)==methods[m]][[1]], Yc, all_classes[c])
        }
      }

      AUC = lapply(all_AUC,function(x){mean(x,na.rm=T)}); AUC2 = lapply(all_AUC2,function(x){mean(x,na.rm=T)}); br1 = lapply(all_br1,function(x){mean(x,na.rm=T)}); br2 = lapply(all_br2,function(x){mean(x,na.rm=T)})
      TPR1 = lapply(all_TPR1,function(x){mean(x,na.rm=T)}); TPR2 = lapply(all_TPR2,function(x){mean(x,na.rm=T)}); FPR1 = lapply(all_FPR1,function(x){mean(x,na.rm=T)}); FPR2 = lapply(all_FPR2,function(x){mean(x,na.rm=T)})
      F11 = lapply(all_F11,function(x){mean(x,na.rm=T)}); F12 = lapply(all_F12,function(x){mean(x,na.rm=T)}); PPV1 = lapply(all_PPV1,function(x){mean(x,na.rm=T)}); PPV2 = lapply(all_PPV2,function(x){mean(x,na.rm=T)})
    }
    ARI1 = lapply(yhats,function(x){as.numeric(mclust::adjustedRandIndex(c(Ys$test$Y), c(x)))})
    ARI2 = lapply(yhats2,function(x){as.numeric(mclust::adjustedRandIndex(c(Ys$test$Y), c(x)))})

    kappa1 = lapply(yhats, function(x){measures::KAPPA(c(Ys$test$Y), c(x))})
    kappa2 = lapply(yhats2, function(x){measures::KAPPA(c(Ys$test$Y), c(x))})

    factors_y = levels(factor(c(Ys$test$Y)))
    mcc1 = lapply(yhats, function(x){yardstick::mcc_vec(truth=factor(c(Ys$test$Y),levels=factors_y), estimate=factor(c(x),levels=factors_y))})
    mcc2 = lapply(yhats2, function(x){yardstick::mcc_vec(truth=factor(c(Ys$test$Y),levels=factors_y), estimate=factor(c(x),levels=factors_y))})
    if(grepl("SIM",dataset)){
      p = lapply(prhats,function(x){abs(x - c(prs$test)[[1]])})
      p2 = lapply(prhats2,function(x){abs(x - c(prs$test)[[1]])})    ### CAT ONLY
      res = list(imp=list(L1s=L1s),
                 inf=list(df=df),
                 pred=list(all_complete=prhats, all_imputed=prhats2, complete=p, imputed=p2,
                           all_AUC_complete=all_AUC, all_AUC_imputed=all_AUC2,
                           all_TPR_complete=all_TPR1, all_TPR_imputed=all_TPR2,
                           all_FPR_complete=all_FPR1, all_FPR_imputed=all_FPR2,
                           all_F1_complete=all_F11, all_F1_imputed=all_F12,
                           AUC_complete=AUC,AUC_imputed=AUC2,
                           TPR_complete=TPR1, TPR_imputed=TPR2,
                           FPR_complete=FPR1, FPR_imputed=FPR2,
                           F1_complete=F11, F1_imputed=F12,
                           PPV_complete=PPV1, PPV_imputed=PPV2,
                           truth=prs,
                           all_Briers1=all_br1, all_Briers2=all_br2,
                           Briers1=br1, Briers2=br2, ARI_complete=ARI1, ARI_imputed=ARI2,
                           kappa1=kappa1, kappa2=kappa2, mcc1=mcc1, mcc2=mcc2))     # output: imputation, inference, prediction
    }else{
      res = list(imp=list(L1s=L1s),
                 inf=NA,
                 pred=list(all_complete=prhats, all_imputed=prhats2,all_AUC_complete=all_AUC, all_AUC_imputed=all_AUC2,
                           AUC_complete=AUC,AUC_imputed=AUC2,all_Briers1=all_br1, all_Briers2=all_br2,
                           all_TPR_complete=all_TPR1, all_TPR_imputed=all_TPR2,
                           all_FPR_complete=all_FPR1, all_FPR_imputed=all_FPR2,
                           all_F1_complete=all_F11, all_F1_imputed=all_F12,
                           TPR_complete=TPR1, TPR_imputed=TPR2,
                           FPR_complete=FPR1, FPR_imputed=FPR2,
                           F1_complete=F11, F1_imputed=F12,
                           PPV_complete=PPV1, PPV_imputed=PPV2,
                           Briers1=br1, Briers2=br2, ARI_complete=ARI1, ARI_imputed=ARI2,
                           kappa1=kappa1, kappa2=kappa2, mcc1=mcc1, mcc2=mcc2))
    }
  } else{

    L1_y = lapply(yhats, function(x){abs(x - Ys$test$Y)})
    L1_y2 = lapply(yhats2, function(x){abs(x - Ys$test$Y)})
    if(grepl("SIM",dataset)){
      res = list(imp=list(L1s=L1s),
                 inf=list(df=df),
                 pred=list(all_complete=yhats, all_imputed=yhats2, complete=L1_y, imputed=L1_y2, truth=Ys))     # output: imputation, inference, prediction
    } else{
      res = list(imp=list(L1s=L1s),
                 inf=NA,
                 pred=list(all_complete=yhats, all_imputed=yhats2, complete=L1_y, imputed=L1_y2, truth=Ys))
    }
  }
  return(res)

}
