#' dlglm: Main wrapper function
#'
#' @param dir_name Directory name where diagnostics and results are saved
#' @param X Matrix of covariates (N x P)
#' @param Y Response variable (N x 1)
#' @param mask_x Mask matrix of X (N x P)
#' @param mask_y Mask vector of Y (N x 1)
#' @param g Vector with entries "train", "valid", or "test" of length N to denote Training-validation-test split partitioning. If the 'test' set is empty, after model training, final imputation is done on the 'train' set. Otherwise, the 'test' set will be imputed. If `g` is not supplied, an 80-20 train-valid split will be generated, and the `train` set will be imputed..
#' @param covars_r_x Vector of 1's and 0's of whether each feature is included as covariates in the missingness model. Need not be specified if `ignorable = T`. Default is using all features as covariates in missingness model. Must be length P (or `ncol(data)`)
#' @param covars_r_y 1 (default) or 0 of whether the response Y is included as a covariate in the missingness model. Need not be specified if `ignorable = T`. Default is using all features as covariates in missingness model.
#' @param learn_r TRUE/FALSE: Whether to learn missingness model via appended NN (TRUE, default), or fit a known logistic regression model (FALSE). If FALSE, `phi0` and `phi` must be specified
#' @param data_types_x Vector of data types ('real', 'count', 'pos', 'cat')
#' @param Ignorable TRUE/FALSE: Whether missingness is ignorable (MCAR/MAR) or nonignorable (MNAR, default). If missingness is known to be ignorable, "ignorable=T" omits missingness model.
#' @param family Family of response Gaussian, Multinomial (generalized Binomial), Poisson (not tested)
#' @param link Link function. Typically: Gaussian - identity, Multinomial - mlogit, Poisson - log
#' @param normalize Pre-normalization of X and Y to mean 0 std 1 (default: FALSE)
#' @param early_stop Early stop criterion based on validation LB to prevent overfitting on training set (default: TRUE)
#' @param trace Output interim training trace information (default: FALSE)
#' @param draw_miss Draw missing values from posterior of missing variable (default: TRUE). Only change to FALSE for debugging purposes
#' @param init_r Initialization scheme of the missingness network ("default" or "alt"). For "alt": missing features are drawn from Unif(-2,2)
#' @param unbalanced If unbalanced categorical variable Y (default: FALSE)
#' @param hyperparams List of grid of hyperparameter values to search. Relevant hyperparameters: `sigma`: activation function ("relu" or "elu"), `h`: number of nodes per hidden layer, `n_hidden_layers`: #hidden layers (except missingness model Decoder_r), `n_hidden_layers_r`: #hidden layers in missingness model (Decoder_r). If "NULL" then set as the same value as each n_hidden_layers (not tuned). Otherwise, can tune a different grid of values; `bs`: batch size, `lr`: learning rate, `dim_z`: dimensionality of latent z, `niw`: number of importance weights (samples drawn from each latent space), `n_imputations`, `n_epochs`: maximum number of epochs
#' @return res object: NIMIWAE fit containing ... on the test set
#' @examples
#' EXAMPLE HERE
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/dlglm}
#'
#' @importFrom reticulate source_python import
#'
#' @export
dlglm = function(dir_name, X, Y, mask_x, mask_y, g, covars_r_x, covars_r_y, learn_r, data_types_x, Ignorable,
                 family, link, normalize, early_stop, trace, draw_miss=T, init_r="default", unbalanced=F,
                 hyperparams = list(sigma="elu", bss=c(1000L), lrs=c(0.01,0.001), impute_bs = 1000L, arch="IWAE",
                                    niws=5L, n_imps = 500L, n_epochss=2002L, n_hidden_layers = c(0L,1L,2L), n_hidden_layers_y = c(0L), n_hidden_layers_r = c(0L,1L),
                                    h=c(128L,64L), h_y=NULL, h_r=c(16L,32L),
                                    dim_zs = c(as.integer(floor(ncol(X)/12)),as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)), as.integer(floor(3*ncol(X)/4))),
                                    L1_weights = 0)){
  ### physionet chose 2, 2, 1 HL's. increase # HL
  # hyperparams = list(sigma="elu", bss=c(1000L), lrs=c(0.01,0.001), impute_bs = 1000L, arch="IWAE",
  #                    niws=5L, n_imps = 500L, n_epochss=2002L,
  #                    # n_hidden_layers = c(0L,1L,2L), n_hidden_layers_y = c(0L), n_hidden_layers_r = c(0L,1L),  # simulations
  #                    n_hidden_layers = c(1L,2L,4L), n_hidden_layers_y = c(0L,1L,2L), n_hidden_layers_r = c(0L,1L),  # Physionet
  #                    h=c(128L,64L), h_y=NULL, h_r=c(16L,32L),
  #                    # dim_zs = as.integer(floor(c(ncol(X)/4, ncol(X)/2, 3*ncol(X)/4))), # simulations
  #                    dim_zs = as.integer(floor(  c(8L, ncol(X)/4, ncol(X)/2, 3*ncol(X)/4)  )),  # Physionet
  #                    L1_weights = 0)){
  ####



  ## Hyperparameters ##
  # # dim_z --> as.integer() does floor()
  # # sigma="elu"; hs=c(64L,128L); bss=c(200L); lrs=c(0.001,0.01); impute_bs = bss[1]; arch="IWAE"
  # # sigma="elu"; hs=c(64L,128L); bss=c(1000L); lrs=c(0.001,0.01); impute_bs = bss[1]; arch="IWAE"   # TEST. COMMENT OUT AND REPLACE W ABOVE LATER
  # # niws=5L; n_epochss=2002L; n_hidden_layers = c(1L, 2L); n_hidden_layers_y = 0L
  # sigma="elu"; hs=c(128L,64L); bss=c(10000L); lrs=c(0.01,0.001); impute_bs = bss[1]; arch="IWAE"   # TEST. COMMENT OUT AND REPLACE W ABOVE LATER
  # niws=5L; n_epochss=2002L; n_hidden_layers = c(1L,2L); n_hidden_layers_y = c(0L,1L); n_hidden_layers_r = c(0L,1L)
  # dim_zs = c(as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)), as.integer(floor(3*ncol(X)/4)))
  # # dim_zs = c(as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)))
  # # if(Ignorable){ L1_weights = 0 } else{ L1_weights = c(1e-1, 5e-2, 0) }
  # L1_weights=0
  sigma = hyperparams$sigma; h = hyperparams$h; h_y = hyperparams$h_y; h_r = hyperparams$h_r; bss = hyperparams$bss; lrs = hyperparams$lrs; impute_bs = hyperparams$impute_bs; arch = hyperparams$arch
  niws = hyperparams$niws; n_imps = hyperparams$n_imps; n_epochss = hyperparams$n_epochss; n_hidden_layers = hyperparams$n_hidden_layers
  n_hidden_layers_y = hyperparams$n_hidden_layers_y; n_hidden_layers_r = hyperparams$n_hidden_layers_r; dim_zs = hyperparams$dim_zs
  L1_weights = hyperparams$L1_weights
  #####################

  if(Ignorable){n_hidden_layers_r = 0L; h_r=0L}

  # (family, link) = (Gaussian, identity), (Multinomial, mlogit), (Poisson, log)

  np = reticulate::import("numpy")
  torch = reticulate::import("torch")
  # reticulate::source_python(system.file("dlglm.py", package = "dlglm"))   # once package is made, put .py in "inst" dir
  reticulate::source_python("dlglm.py")
  P = ncol(X); N = nrow(X)
  P_real = sum(data_types_x=="real"); P_cat = sum(data_types_x=="cat"); P_count = sum(data_types_x=="count"); P_pos = sum(data_types_x=="pos")

  # Transform count data (log) and cat data (subtract by min)
  data_types_x_0 = data_types_x
  # X[,data_types_x=="count"] = log(X[,data_types_x=="count"]+0.001)
  if(P_cat == 0){
    X_aug = X
    mask_x_aug = mask_x
    Cs = np$empty(shape=c(0L,0L))
  } else{
    # reorder to real&count covariates first, then augment cat dummy vars
    X_aug = X[,!(data_types_x %in% c("cat"))]
    mask_x_aug = mask_x[,!(data_types_x %in% c("cat"))]
    covars_r_x_aug = covars_r_x[!(data_types_x %in% c("cat"))]

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
      mask_x_aug = cbind(mask_x_aug, matrix(mask_x[,cat_ids[i]], nrow=N, ncol=Cs[i]))
      covars_r_x_aug = c(covars_r_x_aug, rep(covars_r_x[data_types_x == "cat"][i], Cs[i]) )
    }


    ## column bind real/count and one-hot encoded cat vars
    data_types_x = c( data_types_x[!(data_types_x %in% c("cat"))], rep("cat",sum(Cs)) )
    covars_r_x = covars_r_x_aug
    Cs = np$array(Cs)
  }

  # X[,data_types=="cat"] = X[,data_types=="cat"] - apply(X[,data_types=="cat"],2,min)

  if(family=="Multinomial"){
    levels_Y = levels(factor(Y))   # convert into numeric.
    Y = as.numeric(factor(Y))
    Y = Y-min(Y)
    # Y_aug = matrix(0,nrow=length(Y), ncol=length(unique(Y)))
    # for(i in 1:length(Y)){
    #   Y_aug[i,Y[i]+1] = 1
    # }
    # mask_y_aug = matrix(mask_y,nrow=length(mask_y),ncol=2)
    Y_aug = Y
    mask_y_aug = mask_y
    Cy = length(unique(Y))
  }else{
    levels_Y = NA
    Y_aug = Y
    mask_y_aug = mask_y
    Cy=NULL
  }   # set to start from 0, not 1
  # Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Xs = split(data.frame(X_aug), g)        # split by $train, $test, and $valid
  # Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y_aug), g)        # split by $train, $test, and $valid
  Rxs = split(data.frame(mask_x_aug), g)
  # Rys = split(data.frame(mask_y), g)
  Rys = split(data.frame(mask_y_aug), g)

  # misc fixed params:
  # draw_miss = T
  pre_impute_value = 0
  # n_hidden_layers_r=0  # no hidden layers in decoder_r, no nodes in that hidden layer (doesn't matter what h3 is)
  phi0=NULL; phi=NULL # only input when using logistic regression (known coefs)

  if(normalize){
    norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))
    norm_mean_y=0; norm_sd_y=1
    norm_means_x[data_types_x=="cat"] = 0; norm_sds_x[data_types_x=="cat"] = 1
    # if(family=="Multinomial"){
    #   norm_mean_y=0; norm_sd_y=1
    # } else{
    #   norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
    # }
    dir_name = sprintf("%s/with_normalization",dir_name)
    ifelse(!dir.exists(dir_name),dir.create(dir_name),F)
  }else{
    P_aug = ncol(Xs$train)
    norm_means_x = rep(0, P_aug); norm_sds_x = rep(1, P_aug)
    norm_mean_y = 0; norm_sd_y = 1
  }

  # length_hs = length(h)
  if(is.null(h_y)){ length_hy = 1 } else{ length_hy = length(h_y) }
  if(is.null(h_r)){ length_hr = 1 } else{ length_hr = length(h_r) }

  length_nodes_HLs = if(any(n_hidden_layers==0)){ (length(n_hidden_layers)-1)*length(h)+1 } else { length(n_hidden_layers)*length(h)}
  length_nodes_HLs_r = if(any(n_hidden_layers_r==0)){ (length(n_hidden_layers_r)-1)*length_hr+1 } else { length(n_hidden_layers_r)*length_hr }
  length_nodes_HLs_y = if(any(n_hidden_layers_y==0)){ (length(n_hidden_layers_y)-1)*length_hy+1 } else { length(n_hidden_layers_y)*length_hy }

  n_combs_params = length(bss)*length(lrs)*length(niws)*length(n_epochss)*length_nodes_HLs*length_nodes_HLs_r*length_nodes_HLs_y*length(dim_zs)*length(L1_weights)

  LBs_trainVal = matrix(nrow = n_combs_params,
                        ncol=14 + 5)
  colnames(LBs_trainVal) = c("bs","lr","niw","dim_z", "epochs","nhls","nhl_y","nhl_r","h","h_y","h_r","L1_weight",
                             "LB_train","LB_valid",
                             "MSE_real","MSE_cat","PA_cat","MSE_count","MSE_pos")

  # if(sum(data_types_x=="cat") == 0){
  #   # LBs_trainVal = matrix(nrow = length(bss)*length(lrs)*length(niws)*length(n_epochss)*length(n_hidden_layers)*length(n_hidden_layers_y)*length(n_hidden_layers_r)*length_hs*length(dim_zs)*length(L1_weights),
  #   #                       ncol=15) #ncol = 13)
  #   LBs_trainVal = matrix(nrow = n_combs_params,
  #                         ncol=15) #ncol = 13)
  #   colnames(LBs_trainVal) = c("bs","lr","niw","dim_z", "epochs","nhls","nhl_y","nhl_r","h","h_y","h_r","L1_weight",
  #                              "LB_train",#"MSE_train_x","MSE_train_y",
  #                              "LB_valid",#,"MSE_valid_x","MSE_valid_y"
  #                              "MSE_real"
  #   )
  # }else{
  #   # LBs_trainVal = matrix(nrow = length(bss)*length(lrs)*length(niws)*length(n_epochss)*length(n_hidden_layers)*length(n_hidden_layers_y)*length(n_hidden_layers_r)*length_hs*length(dim_zs)*length(L1_weights),
  #   #                       ncol=17) #ncol = 13)
  #   if(sum(data_types_x=="real") == 0){
  #     LBs_trainVal = matrix(nrow = n_combs_params,
  #                           ncol=16) #ncol = 13)
  #     colnames(LBs_trainVal) = c("bs","lr","niw","dim_z", "epochs","nhls","nhl_y","nhl_r","h","h_y","h_r","L1_weight",
  #                                "LB_train",#"MSE_train_x","MSE_train_y",
  #                                "LB_valid",#,"MSE_valid_x","MSE_valid_y"
  #                                "MSE_cat","PA_cat"
  #     )
  #   }else{
  #     LBs_trainVal = matrix(nrow = n_combs_params,
  #                           ncol=17) #ncol = 13)
  #     colnames(LBs_trainVal) = c("bs","lr","niw","dim_z", "epochs","nhls","nhl_y","nhl_r","h","h_y","h_r","L1_weight",
  #                                "LB_train",#"MSE_train_x","MSE_train_y",
  #                                "LB_valid",#,"MSE_valid_x","MSE_valid_y"
  #                                "MSE_real","MSE_cat","PA_cat"
  #     )
  #   }
  # }
  index = 1

  # i=1;j=1;k=1;m=1;mm=1;nn=1;oo=1
  print("data_types_x"); print(data_types_x)
  print("data_types_x_0"); print(data_types_x_0)
  torch$cuda$empty_cache()

  full_obs_ids = np$array(colSums(mask_x_aug==0)==0)   ## columns missing in train may be diff from in valid/test
  miss_ids = np$array(colSums(mask_x_aug==0)>0)

  for(j in 1:length(bss)){for(k in 1:length(lrs)){
    for(m in 1:length(niws)){for(oo in 1:length(dim_zs)){for(mm in 1:length(n_epochss)){
      for(nn in 1:length(n_hidden_layers)){for(ny in 1:length(n_hidden_layers_y)){for(nr in 1:length(n_hidden_layers_r)){
        ### if #HL = 0, no need to tune h ###
        if(n_hidden_layers[nn]==0){h0=0L} else{h0=h}
        if(n_hidden_layers_r[nr]==0){h_r0=0L} else{h_r0=h_r}
        if(n_hidden_layers_y[ny]==0){h_y0=0L} else{h_y0=h_y}

        for(i in 1:length(h0)){
          if(is.null(h_y)){ h_y0 = h0[i] }    # if h_y specified null, tune same value as h
          if(h_y0==0 & n_hidden_layers_y[ny]>0){ h_y0 = h[1] }    # if nhl_y > 0 but nhl_rest = 0 --> h_y = first nonzero h

          for(ii in 1:length(h_y0)){
            if(is.null(h_r0)){ h_r0 = h0[i] }
            if(h_r0==0 & n_hidden_layers_r[nr]>0){ h_r0 = h[1] }

            for(iii in 1:length(h_r0)){

              for(pp in 1:length(L1_weights)){
                print("bs, lr, niw, n_epochs, nhl, nhl_y, nhl_r, h0, h_y0, h_r0, dim_z, L1_weight")
                print(c(bss[j],lrs[k],niws[m],n_epochss[mm],
                        n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],h0[i],h_y0[ii],h_r0[iii],dim_zs[oo],L1_weights[pp]  ))
                res_train = dlglm(np$array(Xs$train), np$array(Rxs$train), np$array(Ys$train), np$array(Rys$train),
                                  np$array(covars_r_x), np$array(covars_r_y),
                                  np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                                  learn_r, np$array(data_types_x), np$array(data_types_x_0), Cs, Cy,
                                  early_stop, np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),  ########## MIGHT NOT NEED Xs_val...--> may just take Xs$valid
                                  Ignorable, family, link,
                                  impute_bs, arch, draw_miss,
                                  pre_impute_value, n_hidden_layers[nn], n_hidden_layers_y[ny], n_hidden_layers_r[nr],
                                  h0[i], h_y0[ii], h_r0[iii], phi0, phi,
                                  1, NULL, sigma, bss[j], n_epochss[mm],
                                  lrs[k], niws[m], 1L, dim_zs[oo], dir_name=dir_name, trace=trace, save_imps=F, test_temp=0.5, L1_weight=L1_weights[pp], init_r=init_r,
                                  full_obs_ids=full_obs_ids, miss_ids=miss_ids, unbalanced=unbalanced)
                res_valid = dlglm(np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),
                                  np$array(covars_r_x), np$array(covars_r_y),
                                  np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                                  learn_r, np$array(data_types_x), np$array(data_types_x_0), Cs, Cy,
                                  F, NA, NA, NA, NA,
                                  Ignorable, family, link,
                                  impute_bs, arch, draw_miss,
                                  pre_impute_value, n_hidden_layers[nn], n_hidden_layers_y[ny], n_hidden_layers_r[nr],
                                  h0[i], h_y0[ii], h_r0[iii], phi0, phi,
                                  0, res_train$saved_model, sigma, bss[j], 2L,
                                  lrs[k], niws[m], 1L, dim_zs[oo], dir_name=dir_name, trace=trace, save_imps=F, test_temp=res_train$'train_params'$'temp', L1_weight=res_train$'train_params'$'L1_weight', init_r=init_r,
                                  full_obs_ids=full_obs_ids, miss_ids=miss_ids, unbalanced=F)  # no early stopping in validation

                val_LB = res_valid$'LB'    # res_train$'val_LB'

                print(c(bss[j],lrs[k],niws[m],n_epochss[mm],
                        n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],h0[i],h_y0[ii],h_r0[iii],dim_zs[oo],L1_weights[pp],
                        res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                        #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                        val_LB#, res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                        #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                ))
                print(res_valid$'errs')
                LBs_trainVal[index,]=c(bss[j],lrs[k],niws[m],dim_zs[oo],n_epochss[mm],
                                       n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],h0[i],h_y0[ii],h_r0[iii],L1_weights[pp],
                                       res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                                       #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                                       val_LB,
                                       # res_train$'val_LB', #res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                                       #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                                       res_valid$'errs'$'real'$'miss', res_valid$'errs'$'cat0'$'miss', res_valid$'errs'$'cat1'$'miss',
                                       res_valid$'errs'$'count'$'miss', res_valid$'errs'$'pos'$'miss')
                # if(sum(data_types_x=="cat") == 0){
                #   LBs_trainVal[index,]=c(bss[j],lrs[k],niws[m],dim_zs[oo],n_epochss[mm],
                #                          n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],h0[i],h_y0[ii],h_r0[iii],L1_weights[pp],
                #                          res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                #                          #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                #                          val_LB,
                #                          # res_train$'val_LB', #res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                #                          #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                #                          res_valid$'errs'$'real'$'miss'
                #   )
                # }else{
                #   if(sum(data_types_x=="real") == 0){
                #     LBs_trainVal[index,]=c(bss[j],lrs[k],niws[m],dim_zs[oo],n_epochss[mm],
                #                            n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],h0[i],h_y0[ii],h_r0[iii],L1_weights[pp],
                #                            res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                #                            #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                #                            val_LB,
                #                            # res_train$'val_LB', #res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                #                            #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                #                            res_valid$'errs'$'cat0'$'miss', res_valid$'errs'$'cat1'$'miss'
                #     )
                #   } else{
                #     LBs_trainVal[index,]=c(bss[j],lrs[k],niws[m],dim_zs[oo],n_epochss[mm],
                #                            n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],h0[i],h_y0[ii],h_r0[iii],L1_weights[pp],
                #                            res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                #                            #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                #                            val_LB,
                #                            # res_train$'val_LB', #res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                #                            #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                #                            res_valid$'errs'$'real'$'miss', res_valid$'errs'$'cat0'$'miss', res_valid$'errs'$'cat1'$'miss'
                #                            )
                #   }
                # }

                print(LBs_trainVal)

                if(is.na(val_LB)){val_LB=-Inf}

                # save only the best result currently (not all results) --> save memory
                if(index==1){opt_LB = val_LB; save(res_train, file=sprintf("%s/opt_train.out",dir_name)); torch$save(res_train$'saved_model',sprintf("%s/opt_train_saved_model.pth",dir_name))  #; save(opt_train, file="temp_opt_train.out")
                }else if(val_LB > opt_LB){opt_LB = val_LB; save(res_train, file=sprintf("%s/opt_train.out",dir_name)); torch$save(res_train$'saved_model',sprintf("%s/opt_train_saved_model.pth",dir_name))} #; save(opt_train, file="temp_opt_train.out")

                print(paste0("Search grid #",index," of ",n_combs_params))

                rm(res_train)
                rm(res_valid)

                index=index+1

                # release gpu memory
                # reticulate::py_run_string("import torch")
                # reticulate::py_run_string("torch.cuda.empty_cache()")
                torch$cuda$empty_cache()
                gc()
              }}}}}}}}}}}}
  print("Hyperparameter tuning complete")
  saved_model = torch$load(sprintf("%s/opt_train_saved_model.pth",dir_name))
  load(sprintf("%s/opt_train.out",dir_name))
  train_params=res_train$train_params

  test_bs = 100L
  # test_bs = 500L

  res_test = dlglm(np$array(Xs$test), np$array(Rxs$test), np$array(Ys$test), np$array(Rys$test),
                   np$array(covars_r_x), np$array(covars_r_y),
                   np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                   learn_r, np$array(data_types_x), np$array(data_types_x_0), Cs, Cy,
                   F, NA, NA, NA, NA, Ignorable, family, link,
                   test_bs, arch, draw_miss,
                   train_params$pre_impute_value, train_params$n_hidden_layers, train_params$n_hidden_layers_y, train_params$n_hidden_layers_r,
                   train_params$h1, train_params$h2, train_params$h3, phi0, phi,
                   0, saved_model, sigma, test_bs, 2L,
                   train_params$lr, n_imps, 1L, train_params$dim_z, dir_name=dir_name, trace=trace, save_imps=T, test_temp=train_params$'temp', L1_weight=train_params$'L1_weight', init_r=init_r,
                   full_obs_ids=full_obs_ids, miss_ids=miss_ids, unbalanced=F)

  fixed.params = list(dir_name=dir_name, covars_r_x=covars_r_x, covars_r_y=covars_r_y, learn_r=learn_r, Ignorable=Ignorable, family=family, link=link, init_r=init_r, levels_Y=levels_Y)

  return(list(results=res_test, fixed.params=fixed.params))

}
