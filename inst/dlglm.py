def dlglm(X,Rx,Y,Ry, covars_r_x, covars_r_y, norm_means_x, norm_sds_x, norm_mean_y, norm_sd_y, learn_r, data_types_x, data_types_x_0, Cs, Cy, early_stop, X_val, Rx_val, Y_val, Ry_val, Ignorable=False, family="Gaussian", link="identity", impute_bs=None,arch="IWAE",draw_miss=True,pre_impute_value=0,n_hidden_layers=2,n_hidden_layers_y=0,n_hidden_layers_r=0,h1=8,h2=8,h3=0,phi0=None,phi=None,train=1,saved_model=None,sigma="elu",bs = 64,n_epochs = 2002,lr=0.001,niws_z=20,M=20,dim_z=5,dir_name=".",trace=False,save_imps=False, test_temp=0.5, L1_weight=0, init_r="default", full_obs_ids=None, miss_ids=None, unbalanced=False):
  # add early_stop, X_val, Rx_val, Y_val, Ry_val as inputs
  weight_y = 1
  # weight_y = 5
  #family="Gaussian"; link="identity"
  #family="Multinomial"; link="mlogit"
  #family="Poisson"; link="log"
  # covars_r_x: vector of P: 1/0 for inclusion/exclusion of each feature as covariate of missingness model
  # covars_r_y: 1 or 0
  
  ## init_r = "default" or "alt"
  
  # > data_types_x
  # [1] "real"  "real"  "real"  "count" "count" "count" "cat"   "cat"   "cat"   "cat"   "cat"   "cat"
  # > Cs
  # [1] 3 3
  if (h2 is None) and (h3 is None):
    h2=h1; h3=h1
  import torch     # this module not found in Longleaf
  # import torchvision
  import torch.nn as nn
  import numpy as np
  import numpy_indexed as npi
  import scipy.stats
  import scipy.io
  import scipy.sparse
  import pandas as pd
  import torch.distributions as td
  from torch import nn, optim
  from torch.nn import functional as F
  #import torch.nn.utils.prune as prune
  # from torchvision import datasets, transforms
  # from torchvision.utils import save_image
  import time

  from torch.distributions import constraints
  from torch.distributions.distribution import Distribution
  from torch.distributions.utils import broadcast_all
  import torch.nn.functional as F
  from torch.autograd import Variable
  import torch.nn.utils.prune as prune
  from collections import OrderedDict
  import os
  import sys
  import h5py
  os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  
  torch.cuda.empty_cache()
  
  if np.all(Rx==1) and np.all(Ry==1):
    Ignorable=True
  
  # data_types_x = np.repeat("real", X.shape[1])
  ids_real = data_types_x=='real'; p_real=np.sum(ids_real)
  ids_count = data_types_x=='count'; p_count=np.sum(ids_count)
  ids_cat = data_types_x=='cat'; p_cat = len(Cs) #p_cat=np.sum(ids_cat)
  ids_pos = data_types_x=='pos'; p_pos = np.sum(ids_pos)
  
  exists_types = [p_real>0, p_count>0, p_pos>0, p_cat>0]   # real, count, cat types. do they exist?
  print("exists_types (real, count, cat, pos):")
  print(exists_types)
  print("p_real, p_count, p_pos, p_cat:")
  print(str(p_real) + ", " + str(p_count) + ", " + str(p_pos) + ", " + str(p_cat))
  ids_types = [ids_real, ids_count, ids_pos, ids_cat]
  # print("ids_types:")
  # print(ids_types)
  
  

  temp_min=torch.tensor(0.5,device="cuda:0",dtype=torch.float64)
  
  if exists_types[3]:
    temp0 = torch.ones([1], dtype=torch.float64, device='cuda:0')
    temp = torch.ones([1], dtype=torch.float64, device='cuda:0')
  else:
    temp0 = torch.tensor(0.5,device="cuda:0",dtype=torch.float64)  # no cat vars: no need to anneal temperature
    temp = torch.tensor(0.5,device="cuda:0",dtype=torch.float64)
  
  # temp_min=torch.tensor(0.1,device="cuda:0",dtype=torch.float64)
  # ANNEAL_RATE = torch.tensor(0.00003,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  # ANNEAL_RATE = torch.tensor(0.0006,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  ANNEAL_RATE = torch.tensor(0.003,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  # ANNEAL_RATE = torch.tensor(0.01,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  
  
  # niws_z = 1  # only once? --> VAE
  
  #family="Gaussian"   # p(y|x) family
  #link="identity"     # g(E[y|x]) = eta
  if (family=="Multinomial"):
    # C = len(np.unique(Y[~np.isnan(Y)]))   # if we're looking at categorical data, then determine #categories by unique values (nonmissing) in Y
    # print("# classes: " + str(C))
    C=Cy
  else:
    C=1
    

  # Figure out the correct case:
  miss_x = False; miss_y = False
  if np.sum(Rx==0)>0: miss_x = True
  if np.sum(Ry==0)>0: miss_y = True

  #if (not (covars_miss==None).all()):
  #  covars=True
  #  pr1 = np.shape(covars_miss)[1]
  #else:
  #  covars=False
  #  pr1=0
  covars_miss = None ; covars=False  # turned off for now

  # input_r: "r" or "pr" --> what to input into NNs for mask, r (1/0) or p(r=1) (probs)
  # do "r" only for now
  def mse(xhat,xtrue,mask):
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return {'miss':np.mean(np.power(xhat-xtrue,2)[mask<0.5]),'obs':np.mean(np.power(xhat-xtrue,2)[mask>0.5])}
    #return {'miss':np.mean(np.power(xhat-xtrue,2)[~mask]),'obs':np.mean(np.power(xhat-xtrue,2)[mask])}
  def pred_acc(xhat,xtrue,mask,Cs):
    if type(Cs)==int:
      xhat0=xhat; xtrue0=xtrue; mask0=mask
      # xhat0 = np.argmax(xhat, axis=1) + 1
      # xtrue0 = np.argmax(xtrue, axis=1) + 1
      # mask0 = mask[:,0]
    else:
      xhat0 = np.empty([xhat.shape[0], len(Cs)])
      xtrue0 = np.empty([xtrue.shape[0], len(Cs)])
      mask0 = np.empty([mask.shape[0], len(Cs)])
      for i in range(0,len(Cs)):
        # ids = [(Cs[i]*i):(Cs[i]*(i+1))]
        xhat0[:,i] = np.argmax(xhat[:,int(Cs[i]*i):int(Cs[i]*(i+1))], axis=1) + 1
        xtrue0[:,i] = np.argmax(xtrue[:,int(Cs[i]*i):int(Cs[i]*(i+1))], axis=1) + 1
        mask0[:,i] = mask[:,int(Cs[i]*i)]
    # print("xhat0 missing (first 10):")
    # print(xhat0[mask0<0.5][:10])
    # print("xtrue0 missing (first 10):")
    # print(xtrue0[mask0<0.5][:10])
    return {'miss':np.mean((xhat0==xtrue0)[mask0<0.5]),'obs':np.mean((xhat0==xtrue0)[mask0>0.5])}
  ## count X is log transformed --> approximate normal
  ## Turned off: transform before feeding into python script
  # if exists_types[1]:
  #   X[:,ids_count] = np.log( X[:,ids_count] )
  # if exists_types[2]:
  #   where_ids_cat = np.where(ids_cat)
  #   for ii in range(0,p_cat):
  #     X[:,where_ids_cat[ii]] = X[:,where_ids_cat[ii]] - np.min(X[:,where_ids_cat[ii]])   ### transform cat vars to start from 0
  
  
  ## STANDARDIZE X AND Y
  xfull = (X - norm_means_x)/norm_sds_x    # need to not do this if data type is not Gaussian
  if family=="Gaussian":
    yfull = (Y - norm_mean_y)/norm_sd_y
  else: yfull = Y.astype("float")
  
  if early_stop:
    xfull_val = (X_val - norm_means_x)/norm_sds_x
    if family=="Gaussian":
      yfull_val = (Y_val - norm_mean_y)/norm_sd_y
    else: yfull_val = Y_val.astype("float")
  ## OMIT STANDARDIZATION
  # xfull=X; yfull=Y
  
  # Loading and processing data
  n = xfull.shape[0] # number of observations
  p = xfull.shape[1] # number of features
  np.random.seed(1234)

  bs = min(bs,n)
  if (impute_bs==None): impute_bs = n       # if number of observations to feed into imputation, then impute all simultaneously (may be memory-inefficient)
  else: impute_bs = min(impute_bs, n)
  
  xmiss = np.copy(xfull)
  xmiss[Rx==0]=np.nan
  mask_x = np.isfinite(xmiss) # binary mask that indicates which values are missing

  ymiss = np.copy(yfull)
  ymiss[Ry==0]=np.nan
  mask_y = np.isfinite(ymiss)

  yhat_0 = np.copy(ymiss) ### later change this to ymiss with missing y pre-imputed
  xhat_0 = np.copy(xmiss)
  
  if early_stop:
    xmiss_val = np.copy(xfull_val)
    xmiss_val[Rx_val==0]=np.nan
    mask_x_val = np.isfinite(xmiss_val) # binary mask that indicates which values are missing
    
    ymiss_val = np.copy(yfull_val)
    ymiss_val[Ry_val==0]=np.nan
    mask_y_val = np.isfinite(ymiss_val)
    
    yhat_0_val = np.copy(ymiss_val) ### later change this to ymiss with missing y pre-imputed
    xhat_0_val = np.copy(xmiss_val)
  # Custom pre-impute values
  if (pre_impute_value == "mean_obs"):
    xhat_0[Rx==0] = np.mean(xmiss[Rx==1],0); yhat_0[Ry==0] = np.mean(ymiss[Ry==1],0)
    if early_stop: xhat_0_val[Rx_val==0] = np.mean(xmiss_val[Rx_val==1],0); yhat_0_val[Ry_val==0] = np.mean(ymiss_val[Ry_val==1],0)
  elif (pre_impute_value == "mean_miss"):
    xhat_0[Rx==0] = np.mean(xmiss[Rx==0],0); yhat_0[Ry==0] = np.mean(ymiss[Ry==0],0)
    if early_stop: xhat_0_val[Rx_val==0] = np.mean(xmiss_val[Rx_val==0],0); yhat_0_val[Ry_val==0] = np.mean(ymiss_val[Ry_val==0],0)
  elif (pre_impute_value == "truth"):
    xhat_0 = np.copy(xfull); yhat_0 = np.copy(yfull)
    if early_stop: xhat_0_val = np.copy(xfull_val); yhat_0_val = np.copy(yfull_val)
  else:
    xhat_0[np.isnan(xmiss)] = pre_impute_value; yhat_0[np.isnan(ymiss)] = pre_impute_value
    if early_stop: xhat_0_val[np.isnan(xmiss_val)] = pre_impute_value; yhat_0_val[np.isnan(ymiss_val)] = pre_impute_value

  init_mse = mse(xfull,xhat_0,mask_x)
  print("Pre-imputation MSE (obs, should be 0): " + str(init_mse['obs']))
  print("Pre-imputation MSE (miss): " + str(init_mse['miss']))

  prx = np.sum(covars_r_x).astype(int)
  pry = np.sum(covars_r_y).astype(int)
  # pry = C * np.sum(covars_r_y).astype(int)
  pr = prx + pry
  if not learn_r: phi=torch.from_numpy(phi).float().cuda()
  
  # Define decoder/encoder
  if (sigma=="relu"): act_fun=torch.nn.ReLU()
  elif (sigma=="elu"): act_fun=torch.nn.ELU()
  elif (sigma=="tanh"): act_fun=torch.nn.Tanh()
  elif (sigma=="sigmoid"): act_fun=torch.nn.Sigmoid()

  # if train==1:  #### THIS IS NOW DEFINED IN dlglm.R wrapper
  #   ## at test time, full_obs_ids may be different than in training time..
  #   full_obs_ids = np.sum(Rx==0,axis=0)==0    # columns that are fully observed need not have missingness modelled
  #   miss_ids = np.sum(Rx==0,axis=0)>0
    
  
  p_miss = np.sum(~full_obs_ids)
  
  # n_params_xm = 2*p # Gaussian (mean, sd. p features in X)
  if family=="Gaussian":
    n_params_ym = 2
    #n_params_y = 2 # Gaussian (mean, sd. One feature in y)
    n_params_y = 1 # Gaussian (just the mean. learn the SDs (subj specific) directly as parameters like mu_x and sd_x)
  elif family=="Multinomial":
    n_params_ym = C    # probs for each of the K classes
    n_params_y = C
  elif family=="Poisson":
    n_params_ym = 1
    n_params_y = 1
  n_params_r = p_miss*(miss_x) + 1*(miss_y) # Bernoulli (prob. p features in X) --> 1 if missing in y and not X. #of missing features to model.
  # n_params_r = p_miss*(miss_x) + C*(miss_y) # Bernoulli (prob. p features in X) --> 1 if missing in y and not X. #of missing features to model.

  def network_maker(act_fun, n_hidden_layers, in_h, h, out_h, bias=True, dropout=False, init="orthogonal"):
    # dropout=True   ### COMMENT OUT
    # create NN layers
    if n_hidden_layers==0:
      layers = [ nn.Linear(in_h, out_h, bias), ]
    elif n_hidden_layers>0:
      layers = [ nn.Linear(in_h , h, bias), act_fun, ]
      for i in range(n_hidden_layers-1):
        layers.append( nn.Linear(h, h, bias), )
        layers.append( act_fun, )
      layers.append(nn.Linear(h, out_h, bias))
    elif n_hidden_layers<0:
      raise Exception("n_hidden_layers must be >= 0")
    
    # insert dropout layer (if applicable)
    if dropout:
      layers.insert(0, nn.Dropout(p=dropout_pct))
    
    # create NN
    model = nn.Sequential(*layers)
    
    # initialize weights
    def weights_init(layer):
      if init=="normal":
        # if type(layer) == nn.Linear: torch.nn.init.normal_(layer.weight, mean=0, std=1)  # default std.normal
        if type(layer) == nn.Linear: torch.nn.init.normal_(layer.weight, mean=0, std=10)  # default std.normal
      elif init=="orthogonal":
        if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
      elif init=="uniform":
        if type(layer) == nn.Linear: torch.nn.init.uniform_(layer.weight, a=-2, b=2)
    model.apply(weights_init)
    
    return model

  # formulation of NN_xm in mixed data type scenario
  NNs_xm = {}   # include Xo, R (nonignorable), Z, and Yo
  if Ignorable: p2 = p+dim_z+1
  else: p2 = 2*p+dim_z+1
  # if Ignorable: p2 = p+dim_z+C
  # else: p2 = 2*p+dim_z+C
  
  init0 = "orthogonal"   # can change this to "normal", or "uniform" user-defined later
  if miss_x:
    # NN_xm = network_maker(act_fun, n_hidden_layers, 2*p, h1, n_params_xm, True, False).cuda()
    if exists_types[0]: NNs_xm['real'] = network_maker(act_fun, n_hidden_layers, p2, h1, 2*p_real, True, False, init0).cuda()
    if exists_types[1]: NNs_xm['count'] = network_maker(act_fun, n_hidden_layers, p2, h1, 2*p_count, True, False, init0).cuda()
    if exists_types[2]: NNs_xm['pos'] = network_maker(act_fun, n_hidden_layers, p2, h1, 2*p_pos, True, False, init0).cuda()
    if exists_types[3]:
      NNs_xm['cat']=[]
      where_ids_cat = np.where(ids_cat)
      for ii in range(0, p_cat):
        # Cs and NNs_xm['cat'] are lists with elements pertaining to each categorical variable
        # print(ii)
        # print(where_ids_cat[0][ii])
        # print(X[:,where_ids_cat[0][ii]])
        # print(len(np.unique(X[~np.isnan(X[:,where_ids_cat[0][ii]]),where_ids_cat[0][ii]])))
        # Cs.append( len(np.unique(X[~np.isnan(X[:,where_ids_cat[0][ii]]),where_ids_cat[0][ii]])) )
        NNs_xm['cat'].append( network_maker(act_fun, n_hidden_layers, p2, h1, int(Cs[ii]), True, False, init0).cuda() )
  
  # if miss_y: NN_ym = network_maker(act_fun, n_hidden_layers, p+2, h1, n_params_ym, True, False).cuda()
  ## need to fix
  ## NN_ym should condition on Yo, Xm, Xo, and Rx and Ry
  if Ignorable: p3 = p+1
  else: p3 = 2*p+2
  # if Ignorable: p3 = p+2
  # else: p3 = 2*p+2*C
  if miss_y: NN_ym = network_maker(act_fun, n_hidden_layers_y, p3, h1, n_params_ym, True, False, init0).cuda()  # need to fix
  else: NN_ym = None
  # NN_y = network_maker(act_fun, 0, p, h2, n_params_y, False, False).cuda()
  # NN_y = network_maker(act_fun, n_hidden_layers_y, p, h2, n_params_y, True, False).cuda()      # need bias term for nonlinear NN_y!!
  NN_y = network_maker(act_fun, n_hidden_layers_y, p, h2, n_params_y, True, False, init0).cuda()      # need bias term for nonlinear NN_y!!
  # NN_y = network_maker(act_fun, n_hidden_layers_y, p, h2, n_params_y, False, False).cuda()       # no intercept

  # if not Ignorable: NN_r = network_maker(act_fun, n_hidden_layers_r, pr, h3, n_params_r, True, False).cuda()
  if not Ignorable: NN_r = network_maker(act_fun, n_hidden_layers_r, pr, h3, n_params_r, True, False, init0).cuda()
  else: NN_r=None
  
  if init_r=="alt" and not Ignorable:
    dist = torch.distributions.Uniform(torch.Tensor([-2]), torch.Tensor([2]))   # proposed alternative initialization for MNAR/MAR
    # dist = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1]))
    #### sparse not defined here
    # if sparse=="dropout": sh1, sh2 = NN_r[1].weight.shape
    # else: sh1, sh2 = NN_r[0].weight.shape
    sh1, sh2 = NN_r[0].weight.shape
    
    custom_weights = (dist.sample([sh1, sh2]).reshape([sh1,sh2])).cuda()  # N(0,1) or Unif(-2,2)
    with torch.no_grad():
      # if sparse=="dropout": NN_r[1].weight = torch.nn.Parameter(custom_weights)
      # else: NN_r[0].weight = torch.nn.Parameter(custom_weights)
      NN_r[0].weight = torch.nn.Parameter(custom_weights)
  
  # encoder = network_maker(act_fun, n_hidden_layers, p+1, h1, 2*dim_z, True, False).cuda()   # rdeponz = F
  # encoder = network_maker(act_fun, n_hidden_layers, p+C, h1, 2*dim_z, True, False).cuda()   # rdeponz = F
  encoder = network_maker(act_fun, n_hidden_layers, p, h1, 2*dim_z, True, False, init0).cuda()   # rdeponz = F
  # encoder = network_maker(act_fun, n_hidden_layers, 2*p, h1, 2*dim_z, True, False).cuda()  # rdeponz=T
  decoders = { }
  if exists_types[0]:
    # X_real = X[:,ids_real]
    decoders['real'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, 2*p_real, True, False, init0).cuda()
  if exists_types[1]:
    decoders['count'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, 2*p_count, True, False, init0).cuda()
    # decoders['count'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, p_count, True, False).cuda()      # poisson lambda parameter (count p(x))
  if exists_types[2]:
    decoders['pos'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, 2*p_pos, True, False, init0).cuda()
  if exists_types[3]:
    # X_cat = X[:,ids_cat]
    
    decoders['cat']=[]
    for ii in range(0,p_cat):
      # decoders['cat'] are lists with elements pertaining to each categorical variable
      decoders['cat'].append( network_maker(act_fun, n_hidden_layers, dim_z, h1, int(Cs[ii]), True, False, init0).cuda() )
    ###### THIS IS WRONG (needs work): need separate decoder for each categorical covariate, since each covar may have diff # of categories
    # decoder_cat = network_maker(act_fun, n_hidden_layers, dim_z, h1, C*p_cat, True, False).cuda()

  ## Prior p(x): mean and sd for each feature (Uncorrelated covariates)
  # mu_x = torch.zeros(p, requires_grad=True, device="cuda:0"); scale_x = torch.ones(p, requires_grad=True, device="cuda:0")
  
  ## Prior p(x): correlated covariates (unstructured covariance structure)
  # if all_covars:
  #   mu_x = torch.zeros(p, requires_grad=True, device="cuda:0"); scale_x = torch.eye(p, requires_grad=True, device="cuda:0")
  # else:
  #   # only missing covars
  #   mu_x = torch.zeros(p_miss, requires_grad=True, device="cuda:0")
  #   scale_x = torch.eye(p_miss, requires_grad=True, device="cuda:0")
  
  alpha = torch.ones(1, requires_grad=True, device="cuda:0")   # learned directly

  def invlink(link="identity"):
    if link=="identity":
      fx = torch.nn.Identity(0)
    elif link=="log":
      fx = torch.exp
    elif link=="logit":
      fx = torch.nn.Sigmoid()
    elif link=="mlogit":
      fx = torch.nn.Softmax(dim=1)   # same as sigmoid, except imposes that the probs sum to 1 across classes
    return fx
  
  def V(mu, alpha, family="Gaussian"):
    #print(mu.shape)
    if family=="Gaussian":
      out = alpha*torch.ones([mu.shape[0]]).cuda()
    elif family=="Poisson":
      out = mu
    elif family=="NB":
      out = mu + alpha*torch.pow(mu, 2).cuda()
    elif family=="Binomial":
      out = mu*(1-(mu/n_successes))
    elif family=="Multinomial":
      out = mu*(1-mu)
    return out
    
  p_z = td.Independent(td.Normal(loc=torch.zeros(dim_z).cuda(),scale=torch.ones(dim_z).cuda()),1)

  def forward(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, batch_size, niw, temp):
    tiled_iota_x = torch.Tensor.repeat(iota_x,[niw,1])#; tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[niw,1])
    tiled_mask_x = torch.Tensor.repeat(mask_x,[niw,1])#; tiled_tiled_mask_x = torch.Tensor.repeat(tiled_mask_x,[niw,1])
    if not draw_miss: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[niw,1])
    tiled_iota_y = torch.Tensor.repeat(iota_y,[niw,1])#; tiled_tiled_iota_y = torch.Tensor.repeat(tiled_iota_y,[niw,1])
    tiled_mask_y = torch.Tensor.repeat(mask_y,[niw,1])#; tiled_tiled_mask_y = torch.Tensor.repeat(tiled_mask_y,[niw,1])
    if not draw_miss: tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[niw,1])
    
    
    #######################################
    #### p_x via VAE #####
    # out_encoder = encoder(xincluded)
    # print(mask_included.shape)
    # out_encoder = encoder(torch.cat([iota_x, iota_y],1))  # rdeponz = T
    out_encoder = encoder(iota_x)   # rdeponz = T, Z indep on Y given X
    qzgivenx = td.Normal(loc=out_encoder[..., :dim_z],scale=torch.nn.Softplus()(out_encoder[..., dim_z:(2*dim_z)])+0.001)
    params_z = {'mean': out_encoder[..., :dim_z].reshape([batch_size,dim_z]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_encoder[..., dim_z:(2*dim_z)]).reshape([batch_size,dim_z]).detach().cpu().data.numpy()+0.001}

    zgivenx = qzgivenx.rsample([niw]).reshape([-1,dim_z])
    
    # p_x_real = None; p_x_count=None; p_x_cat=None
    
    # initialize lists for categorical variables
    out_decoders = {}; out_decoders['cat'] = []; p_xs={}; p_xs['cat'] = []
    
    params_x = {}; params_x['cat'] = []
    if exists_types[0]:
      out_decoders['real'] = decoders['real'](zgivenx)
      # print(out_decoders['real'])
      p_xs['real'] = td.Normal(loc=out_decoders['real'][..., :p_real],scale=torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)])+0.001)
      params_x['real'] = {'mean': torch.mean(out_decoders['real'][..., :p_real].reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy(),'scale': torch.mean(torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)]).reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy()+0.001}
    if exists_types[1]:
      out_decoders['count'] = decoders['count'](zgivenx)
      # p_x_count = td.Poisson(rate=out_decoder_count[..., :p_count])    # no log transformation of data
      p_xs['count'] = td.Normal(loc=out_decoders['count'][..., :p_count],scale=torch.nn.Softplus()(out_decoders['count'][..., p_count:(2*p_count)])+0.001)
      params_x['count'] = {'mean': torch.mean(out_decoders['count'][..., :p_count].reshape([niw,batch_size,p_count]),0).detach().cpu().data.numpy(),'scale': torch.mean(torch.nn.Softplus()(out_decoders['count'][..., p_count:(2*p_count)]).reshape([niw,batch_size,p_count]),0).detach().cpu().data.numpy()+0.001}
      # p_xs['count'] = td.Poisson(rate=torch.nn.Softplus()(out_decoders['count'][..., :p_count])+0.001)    # no log transformation of data
      # params_x['count'] = {'lambda': torch.mean(torch.nn.Softplus()(out_decoders['count'][..., :p_count]).reshape([niw, batch_size,p_count]),0).detach().cpu().data.numpy()+0.001}
    if exists_types[2]:
      out_decoders['pos'] = decoders['pos'](zgivenx)
      # print(out_decoders['real'])
      p_xs['pos'] = td.LogNormal(loc=out_decoders['pos'][..., :p_pos],scale=torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)])+0.001)
      params_x['pos'] = {'mean': torch.mean(out_decoders['pos'][..., :p_pos].reshape([niw,batch_size,p_pos]),0).detach().cpu().data.numpy(),'scale': torch.mean(torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)]).reshape([niw,batch_size,p_pos]),0).detach().cpu().data.numpy()+0.001}
    if exists_types[3]:
      for ii in range(0,p_cat):
        out_decoders['cat'].append( torch.clamp(torch.nn.Softmax(dim=1)( decoders['cat'][ii](zgivenx) ), min=0.0001, max=0.9999).reshape([niw,batch_size,-1]).reshape([niw*batch_size,-1]) )
        p_xs['cat'].append( td.RelaxedOneHotCategorical(temperature=temp, probs = out_decoders['cat'][ii]) )
        
        params_x['cat'].append(torch.mean(out_decoders['cat'][ii].reshape([niw,batch_size,-1]),0).detach().cpu().data.numpy())

    
    xm_flat = torch.Tensor.repeat(iota_x,[niw,1])
    ym_flat = torch.Tensor.repeat(iota_y,[niw,1])

    #################################################################################
    #################################################################################
    ############################ NN_xms HERE ########################################
    #################################################################################
    #################################################################################
    
    ## NN_xm ## q(xm|xo,r)    (if missing in x detected)
    if miss_x:
      # out_NN_xm = NN_xm(torch.cat([iota_x,mask_x],1))
      # # bs x p -- > sample niw times
      # qxmgivenxor = td.Normal(loc=out_NN_xm[..., :p],scale=torch.nn.Softplus()(out_NN_xm[..., p:(2*p)])+0.001)    ### condition contribution of this term in the ELBO by miss_x
      # params_xm = {'mean':out_NN_xm[..., :p], 'scale':torch.nn.Softplus()(out_NN_xm[..., p:(2*p)])+0.001}
      # if draw_miss: xm = qxmgivenxor.rsample([niw]); xm_flat = xm.reshape([niw*batch_size,p])
      
      # initialize lists for categorical variables
      outs_NN_xm = {}; outs_NN_xm['cat'] = []; qxmgivenxors = {}; qxmgivenxors['cat'] = []
      params_xm = {}; params_xm['cat'] = []
    
      if exists_types[0]:
        if Ignorable:   outs_NN_xm['real'] = NNs_xm['real'](torch.cat([tiled_iota_x,zgivenx,tiled_iota_y],1))
        else:  outs_NN_xm['real'] = NNs_xm['real'](torch.cat([tiled_iota_x,tiled_mask_x,zgivenx,tiled_iota_y],1))
        qxmgivenxors['real'] = td.Normal(loc=outs_NN_xm['real'][..., :p_real],scale=torch.nn.Softplus()(outs_NN_xm['real'][..., p_real:(2*p_real)])+0.001)
        params_xm['real'] = {'mean': torch.mean(outs_NN_xm['real'][..., :p_real].reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy(),'scale': torch.mean(torch.nn.Softplus()(outs_NN_xm['real'][..., p_real:(2*p_real)]).reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy()+0.001}
      if exists_types[1]:
        if Ignorable:  outs_NN_xm['count'] = NNs_xm['count'](torch.cat([tiled_iota_x,zgivenx,tiled_iota_y],1))
        else:  outs_NN_xm['count'] = NNs_xm['count'](torch.cat([tiled_iota_x,tiled_mask_x,zgivenx,tiled_iota_y],1))
        qxmgivenxors['count'] = td.Normal(loc=outs_NN_xm['count'][..., :p_count],scale=torch.nn.Softplus()(outs_NN_xm['count'][..., p_count:(2*p_count)])+0.001)   # log transformed count data
        params_xm['count'] = {'mean': torch.mean(outs_NN_xm['count'][..., :p_count].reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy(),'scale': torch.mean(torch.nn.Softplus()(outs_NN_xm['count'][..., p_count:(2*p_count)]).reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy()+0.001}
      if exists_types[2]:
        if Ignorable:   outs_NN_xm['pos'] = NNs_xm['pos'](torch.cat([tiled_iota_x,zgivenx,tiled_iota_y],1))
        else:  outs_NN_xm['pos'] = NNs_xm['pos'](torch.cat([tiled_iota_x,tiled_mask_x,zgivenx,tiled_iota_y],1))
        qxmgivenxors['pos'] = td.LogNormal(loc=outs_NN_xm['pos'][..., :p_pos],scale=torch.nn.Softplus()(outs_NN_xm['pos'][..., p_pos:(2*p_pos)])+0.001)   # log transformed count data
        params_xm['pos'] = {'mean': torch.mean(outs_NN_xm['pos'][..., :p_pos].reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy(),'scale': torch.mean(torch.nn.Softplus()(outs_NN_xm['pos'][..., p_pos:(2*p_pos)]).reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy()+0.001}
      if exists_types[3]:
        for ii in range(0,p_cat):
          if Ignorable:  outs_NN_xm['cat'].append( torch.clamp( torch.nn.Softmax(dim=1)( NNs_xm['cat'][ii](torch.cat([tiled_iota_x,zgivenx,tiled_iota_y],1)) ), min=0.0001, max=0.9999))
          else:  outs_NN_xm['cat'].append( torch.clamp( torch.nn.Softmax(dim=1)( NNs_xm['cat'][ii](torch.cat([tiled_iota_x,tiled_mask_x,zgivenx,tiled_iota_y],1)) ), min=0.0001, max=0.9999))
          
          qxmgivenxors['cat'].append( td.RelaxedOneHotCategorical(temperature=temp, probs = outs_NN_xm['cat'][ii]) )
          params_xm['cat'].append(torch.mean(outs_NN_xm['cat'][ii].reshape([niw,batch_size,-1]),0).detach().cpu().data.numpy())
      
      xm_flat = torch.zeros([M*niw*batch_size, p]).cuda()
      # xm_flat_cat = []
      if draw_miss:
        # need to sample real, count, cat data and concatenate
        if exists_types[0]: xm_flat[:,ids_real] = qxmgivenxors['real'].rsample([M]).reshape([M*niw*batch_size,-1])
        if exists_types[1]:
          xm_flat[:,ids_count] = qxmgivenxors['count'].rsample([M]).reshape([M*niw*batch_size,-1])
          # xm_flat[:,ids_count] = torch.exp( qxmgivenxors['count'].rsample([M]).reshape([M*batch_size,-1]) )     # normal samples --> exp() to better approximate original counts
        if exists_types[2]: xm_flat[:,ids_pos] = qxmgivenxors['pos'].rsample([M]).reshape([M*niw*batch_size,-1])
        if exists_types[3]: 
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            
            # if Ignorable: xm_flat[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)] = qxmgivenxors['cat'][ii].sample([M]).reshape([M*niw*batch_size,-1])
            # else: xm_flat[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)] = qxmgivenxors['cat'][ii].rsample([M]).reshape([M*niw*batch_size,-1])
            xm_flat[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)] = qxmgivenxors['cat'][ii].rsample([M]).reshape([M*niw*batch_size,-1])
        # if torch.sum(torch.isnan(xm_flat))>0:
        #   print("minibatched data:")
        #   print(iota_x[:1])
        #   print(iota_x.shape)
        #   print("mask:")
        #   print(mask_x[:1])
        #   print("p_xs['real'] (mean/scale):")
        #   print(params_x['real']['mean'][:4])
        #   print(params_x['real']['scale'][:4])
        #   print("xm_flat:")
        #   print(xm_flat[:4])
        #   sys.exit("NA in xm_flat")
      else: xm_flat = tiled_iota_xfull   # MAY NEED ADJUSTMENT IF CATEGORICAL VARIABLES (onehot)
    else: 
      qxmgivenxors=None; params_xm=None; xm_flat = torch.Tensor.repeat(iota_x,[M,1])
    # organize completed (sampled) xincluded for missingness model. observed values are not sampled
    if miss_x:  # commented out if miss_y: because M is usually set to 1, i.e. 1 imputation per sample of Z (niw or L)
      # if miss_y:
      #   tiled_xm_flat = torch.Tensor.repeat(xm_flat,[M,1])
      #   xincluded = tiled_tiled_iota_x*(tiled_tiled_mask_x) + tiled_xm_flat*(1-tiled_tiled_mask_x)
      #   mask_included=tiled_tiled_mask_x
      # else:
      xincluded = tiled_iota_x*(tiled_mask_x) + xm_flat*(1-tiled_mask_x)
      mask_included=tiled_mask_x
    else:
      xincluded = iota_x
      mask_included=tiled_mask_x
    
    # print("before:")
    # print(xincluded[:,ids_cat])
    xincluded[:,ids_cat] = torch.clamp(xincluded[:,ids_cat], min=0.0001, max=0.9999)
    
    # xincluded0 = torch.clone(xincluded)   # untransformed counts
    
    # xincluded[:,ids_count] = torch.exp(xincluded[:,ids_count])
    # print("after:")
    # print(xincluded[:,ids_cat])
    
    # print("xincluded shape:")
    # print(xincluded.shape)
    

    ## NN_ym ## p(ym|yo,x,rx, ry, (z??))   (if missing in y detected)    
    if miss_y:
      if not miss_x:
        if Ignorable:  out_NN_ym = NN_ym(torch.cat([iota_y, iota_x],1))
        else:   out_NN_ym = NN_ym(torch.cat([iota_y, iota_x, mask_x, mask_y],1))
        # bs x 1 --> sample niw times
      elif miss_x:
        if Ignorable:   out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiled_mask_x*tiled_iota_x + (1-tiled_mask_x)*xm_flat],1))
        else:   out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiled_mask_x*tiled_iota_x + (1-tiled_mask_x)*xm_flat, tiled_mask_x, tiled_mask_y],1))
        # (niw*bs) x 1 --> sampled niw times
      if family=="Gaussian":
        qymgivenyor = td.Normal(loc=out_NN_ym[..., :1],scale=torch.nn.Softplus()(out_NN_ym[..., 1:2])+0.001)     ### condition contribution of this term in the ELBO by miss_y
        params_ym = {'mean':torch.mean(torch.mean(out_NN_ym[..., :1].reshape([niw, M**(int(miss_x)),batch_size,1]),0),0).reshape([batch_size,1]).detach().cpu().data.numpy(), 'scale':torch.mean(torch.mean(torch.nn.Softplus()(out_NN_ym[..., 1:2]).reshape([niw, M**(int(miss_x)),batch_size,1]),0),0).reshape([batch_size,1]).detach().cpu().data.numpy()+0.001}
      elif family=="Multinomial":
        qymgivenyor = td.RelaxedOneHotCategorical(temperature=temp, probs = invlink(link)(out_NN_ym))
        params_ym = {'probs':torch.mean(out_NN_ym.reshape([niw,batch_size,-1]),0).detach().cpu().data.numpy()}
      if draw_miss: ym = qymgivenyor.rsample([niw]); ym_flat = ym.reshape([-1,1])    # ym_flat is (niw*bs x 1) if no miss_x, and (niw*niw*bs x 1) if miss_x
      else: ym = tiled_iota_yfull; ym_flat = ym.reshape([-1,1])
    else:
      qymgivenyor=None; params_ym=None; ym_flat = torch.Tensor.repeat(iota_y,[niw,1])
    
    # organize completed (sampled) xincluded for missingness model. observed values are not sampled
    if miss_y:
      # if miss_x:  yincluded = tiled_tiled_iota_y*(tiled_tiled_mask_y) + ym_flat*(1-tiled_tiled_mask_y)
      # else:
      yincluded = tiled_iota_y*(tiled_mask_y) + ym_flat*(1-tiled_mask_y)
    else:
      # if miss_x:  yincluded = tiled_iota_y
      # else:  yincluded = iota_y
      yincluded = tiled_iota_y
    ## NN_y ##      p(y|x)
    out_NN_y = NN_y(xincluded)     # if miss_x and miss_y: this becomes niw*niw*bs x p, otherwise: niw*bs x p
    if family=="Gaussian":
      mu_y = invlink(link)(out_NN_y[..., 0]);  var_y = V(mu_y, torch.nn.Softplus()(alpha)+0.001, family)   # default: link="identity", family="Gaussian"
      pygivenx = td.Normal(loc = mu_y, scale = (var_y)**(1/2))    # scale = sd = var^(1/2)
      params_y = {'mean': torch.mean(torch.mean(mu_y.reshape([niw, M**(int(miss_x)),batch_size,1]),0),0).reshape([batch_size,1]).detach().cpu().data.numpy(), 'scale': (torch.mean(torch.mean(var_y.reshape([niw, M**(int(miss_x)),batch_size,1]),0),0).reshape([batch_size,1]).detach().cpu().data.numpy())**(1/2)}
      # params_y = {'mean': torch.mean(mu_y.reshape([niw,batch_size,1]),0).reshape([batch_size,1]).detach().cpu().data.numpy(), 'scale': (torch.mean(var_y.reshape([niw, batch_size,1]),0).reshape([batch_size,1]).detach().cpu().data.numpy())**(1/2)}
    elif family=="Multinomial":
      probs = invlink(link)(out_NN_y)
      pygivenx = td.OneHotCategorical(probs=probs)
      #print("probs:"); print(probs)
      #print("pygivenx (event_shape):"); print(pygivenx.event_shape)
      #print("pygivenx (batch_shape):"); print(pygivenx.batch_shape)
      params_y = {'probs': torch.mean(torch.mean(probs.reshape([niw, M**(int(miss_x)),batch_size,C]),0),0).reshape([batch_size,C]).detach().cpu().data.numpy()}
      # params_y = {'probs': torch.mean(probs.reshape([niw, batch_size, C]),0).reshape([batch_size,C]).detach().cpu().data.numpy()}
    elif family=="Poisson":
      lambda_y = invlink(link)(out_NN_y[..., 0])  # variance is the same as mean in Poisson
      pygivenx = td.Poisson(rate = lambda_y)
      #### PROBLEM HERE. IF WE'RE ALLOWING FOR miss_y, then Poisson family can't be possible. Either a priori transform --> Normal or no transform --> count
      params_y = {'lambda': torch.mean(torch.mean(lambda_y.reshape([niw, M**(int(miss_x)),batch_size,1]),0),0).reshape([batch_size,1]).detach().cpu().data.numpy()}
      # params_y = {'lambda': torch.mean(lambda_y.reshape([niw, batch_size,1]),0).reshape([batch_size,1]).detach().cpu().data.numpy()}

    #print(pygivenx.rsample().shape)

    ## NN_r ##   p(r|x,y,covars): always. Include option to specify covariates in X, y, and additional covars_miss
    # Organize covariates for missingness model (NN_r)
    if covars_r_y==1:
      # print(xincluded[:,covars_r_x==1].shape)
      # print(yincluded.shape)
      if np.sum(covars_r_x)>0: covars_included = torch.cat([xincluded[:,covars_r_x==1], yincluded],1)
      else: covars_included = yincluded
    elif covars_r_y==0:
      if np.sum(covars_r_x)>0: covars_included = xincluded[:,covars_r_x==1]
      # else: IGNORABLE HERE. NO COVARIATES
    
    #print(covars_included.shape)
    #print(NN_r)
    if not Ignorable:
      if (covars): out_NN_r = NN_r(torch.cat([covars_included, covars_miss],1))   # right now: just X in as covariates (Case 1)    (niw*niw*bs x p) for case 3, (niw*bs x p) for other cases
      else: out_NN_r = NN_r(covars_included)        # can additionally include covariates
      prgivenxy = td.Bernoulli(logits = out_NN_r)   # for just the features with missing valuess
      params_r = {'probs': torch.mean(torch.mean(torch.mean(torch.nn.Sigmoid()(out_NN_r).reshape([M**(int(miss_y)),niw, M**(int(miss_x)),batch_size,n_params_r]),0),0),0).reshape([batch_size,n_params_r]).detach().cpu().data.numpy()}
    else: prgivenxy=None; params_r=None


    # return xincluded, yincluded, p_x, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_y, params_r
    return xincluded, yincluded, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx
  # return xincluded, yincluded, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx
  
  # cases (1): miss_x, (2) miss_y, (3) miss_x and miss_y
  # xincluded: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
  # yincluded: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
  # p_xs: 0 x p (broken into p_x_real, p_x_count, p_xs_cat)
  # qxmgivenxors: (1) bs x p, (2) None, (3) bs x p
  # qymgivenyor: (1) None, (2) bs x p, (3) (niw*bs) x p
  # pygivenx: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
  # prgivenxy: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
   
  def compute_loss(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, covar_miss, temp):
    batch_size = iota_x.shape[0]
    tiled_iota_x = torch.Tensor.repeat(iota_x,[niws_z,1]).cuda(); tiled_iota_y = torch.Tensor.repeat(iota_y,[niws_z,1]).cuda()
    #tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[niws_z,1]).cuda(); tiled_tiled_iota_y = torch.Tensor.repeat(tiled_iota_y,[niws_z,1]).cuda()
    tiled_mask_x = torch.Tensor.repeat(mask_x,[niws_z,1]).cuda()#; tiled_tiled_mask_x = torch.Tensor.repeat(tiled_mask_x,[niws_z,1]).cuda()
    tiled_mask_y = torch.Tensor.repeat(mask_y,[niws_z,1]).cuda()#; tiled_tiled_mask_y = torch.Tensor.repeat(tiled_mask_y,[niws_z,1]).cuda()
    if not draw_miss: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[niws_z,1]).cuda(); tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[niws_z,1]).cuda()
    else: tiled_iota_xfull = None
    
    if covars: tiled_covars_miss = torch.Tensor.repeat(covar_miss,[M,1])
    else: tiled_covars_miss=None

    xincluded, yincluded, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx = forward(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, batch_size, niws_z, temp)
    
    if family=="Multinomial":
      yincluded=torch.nn.functional.one_hot(yincluded.to(torch.int64),num_classes=C).reshape([-1,C])

    # form of ELBO: log p(y|x) + log p(x) + log p(r|x) - log q(xm|xo, r)
    
    logpx_real = torch.Tensor([0]).cuda(); logpx_cat = torch.Tensor([0]).cuda(); logpx_count = torch.Tensor([0]).cuda(); logpx_pos = torch.Tensor([0]).cuda()
    logqxmgivenxor_real = torch.Tensor([0]).cuda(); logqxmgivenxor_cat = torch.Tensor([0]).cuda(); logqxmgivenxor_count = torch.Tensor([0]).cuda(); logqxmgivenxor_pos = torch.Tensor([0]).cuda()
    
    ## COMPUTE LOG PROBABILITIES ##
    if miss_x and miss_y:     # case 3
      # log p(r|x,y) # niw*niw*bs x p
      if not Ignorable:
        all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_mask_x[:,miss_ids], tiled_mask_y],1))
        logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M*M,batch_size])
      else: all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda()
      # log p(y|x)   # niw*niw*bs x p
      if family=="Gaussian": all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))    #yincluded: M*M*batch_size
      else: all_log_pygivenx = pygivenx.log_prob(yincluded)

      logpygivenx = all_log_pygivenx.reshape([niws_z*M*M,batch_size])
      
      ## log p(x) with VAE:
      if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M*M,batch_size])
      if exists_types[1]:
        logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M*M,batch_size])
        # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count])),axis=1).reshape([M*M,batch_size])
      if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M*M,batch_size])
      if exists_types[3]:
        # logpx_cat = torch.sum(p_x_cat.log_prob(xincluded[:,ids_cat]),axis=1).reshape([M*M,batch_size])
        for ii in range(0,p_cat):
          if ii==0: C0=0; C1=int(Cs[ii])
          else: C0=C1; C1=C0 + int(Cs[ii])
          
          if ii==0: logpx_cat = torch.sum(p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])   ),axis=1).reshape([-1,batch_size])
          else: logpx_cat = logpx_cat + torch.sum(p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])   ),axis=1).reshape([-1,batch_size])

      ## log q(xm|xo,r)
      # logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M*M,batch_size,p])).reshape([M*M*batch_size,p])*(1-tiled_mask_x),1).reshape([M*M,batch_size])
      if exists_types[0]: logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob( xincluded[:,ids_real].reshape([-1,p_real]) )*(1-tiled_mask_x[:,ids_real]),1).reshape([-1,batch_size])
      if exists_types[1]: logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([-1,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M*M,batch_size])
      if exists_types[2]: logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob( xincluded[:,ids_pos].reshape([-1,p_pos]) )*(1-tiled_mask_x[:,ids_pos]),1).reshape([-1,batch_size])
      if exists_types[3]:
        # logpx_cat = torch.sum(p_x_cat.log_prob(xincluded[:,ids_cat]),axis=1).reshape([M*M,batch_size])
        for ii in range(0,p_cat):
          if ii==0: C0=0; C1=int(Cs[ii])
          else: C0=C1; C1=C0 + int(Cs[ii])
          
          if ii==0: logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob(  xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])  )*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
          else: logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])   )*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

      logpxsum = torch.zeros([M*M,batch_size]).cuda(); logqxsum = torch.zeros([niws_z*M*M,batch_size]).cuda()
      
      # log q(ym|yo,r,xm,xo)
      logqymgivenyor = (qymgivenyor.log_prob(yincluded)*(1-tiled_mask_y)).reshape([niws_z*M*M,batch_size])
    else:
      # log p(r|x,y)
      if miss_x and not miss_y:  # case 1
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiled_mask_x[:,miss_ids])  # M*bs x p_miss
          logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M,batch_size])
        else: all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda()
        ## log q(xm|xo,r)
        # logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M,batch_size,-1])).reshape([M*batch_size,-1])*(1-tiled_mask_x),1).reshape([M,batch_size]); logqymgivenyor=0
        if exists_types[0]: logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob(xincluded[:,ids_real].reshape([niws_z*M*batch_size,p_real]))*(1-tiled_mask_x[:,ids_real]),1).reshape([niws_z*M,batch_size])
        if exists_types[1]:
          logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([niws_z*M*batch_size,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M,batch_size])
          # logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(torch.log(xincluded[:,ids_count]).reshape([M*M,batch_size,p_count])).reshape([M*M*batch_size,p_count])*(1-tiled_mask_x[:,ids_count]),1).reshape([M*M,batch_size])
        if exists_types[2]: logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob(xincluded[:,ids_pos].reshape([niws_z*M*batch_size,p_pos]))*(1-tiled_mask_x[:,ids_pos]),1).reshape([niws_z*M,batch_size])
        if exists_types[3]:
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([niws_z*M*batch_size,int(Cs[ii])]) )  *  (1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
            else: logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([niws_z*M*batch_size,int(Cs[ii])]) )  *  (1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

        # log p(y|x)   # all cases set to be (niw*niw*bs) x p
        # print(pygivenx.event_shape)
        # print(pygivenx.batch_shape)
        # print(yincluded.shape)
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1])) #yincluded: M*batch_size

        #print("Diagnostics:"); print(yincluded.shape); print(pygivenx.event_shape); print(pygivenx.batch_shape); print(all_log_pygivenx.shape)
        logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])
        
        ## log p(x): uncorrelated
        # logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        ## log p(x): correlated (unstructured)
        # just missing covars
        # logpx = p_x.log_prob(xincluded[:,miss_ids]).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        
        ## log p(x) with VAE:
        # print(xincluded[:,ids_real].shape)
        # print(torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real].reshape([niws_z*M,batch_size,p_real])),axis=2).shape)
        if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[1]:
          logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
          # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count]).reshape([M,batch_size,p_count])),axis=2).reshape([M,batch_size])
        if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[3]:
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logpx_cat = p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)] ).reshape([niws_z*M,batch_size])
            else: logpx_cat = logpx_cat + p_xs['cat'][ii].log_prob(  xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)]  ).reshape([niws_z*M,batch_size])
        

        logpxsum = torch.zeros([niws_z*M,batch_size]).cuda(); logqxsum = torch.zeros([niws_z*M,batch_size]).cuda()
      elif not miss_x and miss_y: # case 2
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiled_mask_y)
          logprgivenxy = all_logprgivenxy.reshape([niws_z*M,batch_size])   # no need to sum across columns (missingness of just y)
        else: all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda()
        # log q(ym|xo,r)
        logqymgivenyor = (qymgivenyor.log_prob(yincluded)*(1-tiled_mask_y)).reshape([niws_z*M,batch_size]); logqxsum=0
        # log p(y|x)   # all cases set to be (niw*niw*bs) x p
        
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])
        ## log p(x): uncorrelated
        # logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        ## log p(x): correlated (unstructured)

        # just missing covars
        # logpx = p_x.log_prob(xincluded[:,miss_ids]).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        
        ## log p(x) with VAE:
        if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[1]: 
          logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
          # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count])),axis=1).reshape([M,batch_size])
        if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[3]: 
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logpx_cat = (p_xs['cat'].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C[ii]])   )).reshape([-1,batch_size])
            else: logpx_cat = logpx_cat + (p_xs['cat'].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C[ii]]) )).reshape([-1,batch_size])

        logpxsum = torch.zeros([M,batch_size]).cuda()
      else:     # no missing
        all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda(); logqxsum=torch.Tensor([0]).cuda(); logqymgivenyor=torch.Tensor([0]).cuda()

        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([niws_z*1*batch_size])
        ## log p(x): uncorrelated
        # logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([1,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        ## log p(x): correlated (unstructured)

        ## just missing covars
        # logpx = p_x.log_prob(xincluded[:,miss_ids]).reshape([1,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        
        ## log p(x) with VAE:
        if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*1*batch_size])
        if exists_types[1]:
          logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*1*batch_size])
          # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count])),axis=1).reshape([1,batch_size])
        if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*1*batch_size])
        if exists_types[3]:
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logpx_cat = (p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C])   )).reshape([-1])
            else: logpx_cat = logpx_cat + (p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C])   )).reshape([-1])

        logpxsum = torch.zeros([niws_z*1,batch_size]).cuda(); logqxsum = torch.zeros([niws_z*1*batch_size]).cuda()

      
      ##logpx = torch.sum(p_x.log_prob(xincluded)*(1-tiled_mask_x),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (just missing x)
    
    if not Ignorable: sum_logpr = np.sum(logprgivenxy.cpu().data.numpy())
    else: sum_logpr = 0

    sum_logpygivenx = np.sum(logpygivenx.cpu().data.numpy())
    # sum_logpx = np.sum(logpx.cpu().data.numpy())
    sum_logpx = 0
    if exists_types[0]: sum_logpx = sum_logpx + np.sum(logpx_real.cpu().data.numpy())
    if exists_types[1]: sum_logpx = sum_logpx + np.sum(logpx_count.cpu().data.numpy())
    if exists_types[2]: sum_logpx = sum_logpx + np.sum(logpx_pos.cpu().data.numpy())
    if exists_types[3]: sum_logpx = sum_logpx + np.sum(logpx_cat.cpu().data.numpy())

    if miss_y: sum_logqym = np.sum(logqymgivenyor.cpu().data.numpy())
    else: sum_logqym = 0; logqymgivenyor=torch.Tensor([0]).cuda()

    #c = torch.cuda.memory_cached(0); a = torch.cuda.memory_allocated(0)
    #print("memory free:"); print(c-a)  # free inside cache


    # log q(ym|yo,r) ## add this into ELBO too
    # Case 1: x miss, y obs --> K samples of X
    # Case 2: x obs, y miss --> K samples of Y
    # Case 3: x miss, y miss --> K samples of X, K*M samples of Y. NEED TO MAKE THIS CONSISTENT: just make K=M and M samples of X and M samples of Y
    
    #initialized these sum terms earlier
    if exists_types[0]: logpxsum = logpxsum + logpx_real; logqxsum = logqxsum + logqxmgivenxor_real
    if exists_types[1]: logpxsum = logpxsum + logpx_count; logqxsum = logqxsum + logqxmgivenxor_count
    if exists_types[2]: logpxsum = logpxsum + logpx_pos; logqxsum = logqxsum + logqxmgivenxor_pos
    if exists_types[3]: logpxsum = logpxsum + logpx_cat; logqxsum = logqxsum + logqxmgivenxor_cat

    logpz = p_z.log_prob(zgivenx).reshape([-1,batch_size])
    # print(logpz.shape)
    # print(zgivenx.shape)
    # print(qzgivenx.event_shape)
    # print(qzgivenx.batch_shape)
    logqzgivenx = torch.sum(qzgivenx.log_prob(zgivenx.reshape([-1,batch_size,dim_z])),axis=2).reshape([-1,batch_size])
    
    # print(zgivenx.shape)
    # print(logpz.shape)
    # print(logqzgivenx.shape)
    
    if arch=="VAE":
      ## VAE NEGATIVE LOG-LIKE ##
      # neg_bound = -torch.mean(logpygivenx + logpx + logprgivenxy - logqxmgivenxor - logqymgivenyor)
      # neg_bound = -torch.mean(logpygivenx + logpxsum + logprgivenxy - logqxmgivenxor - logqymgivenyor)
      neg_bound = -torch.sum(weight_y*logpygivenx + logpxsum + logprgivenxy - logqxsum - logqymgivenyor)
    elif arch=="IWAE":
      ## IWAE NEGATIVE LOG-LIKE ##
      # neg_bound = np.log(M) + np.log(M)*(miss_x and miss_y) - torch.mean(torch.logsumexp(logpygivenx + logpx + logprgivenxy - logqxmgivenxor - logqymgivenyor,0))
      # neg_bound = np.log(M) + np.log(M)*(miss_x and miss_y) - torch.mean(torch.logsumexp(logpygivenx + logpxsum + logpz + logprgivenxy - logqzgivenx - logqxmgivenxor - logqymgivenyor,0))
      neg_bound = -torch.sum(torch.logsumexp(weight_y*logpygivenx + logpxsum + logpz + logprgivenxy - logqzgivenx - logqxsum - logqymgivenyor,0))
    
    # print("logpygivenx")
    # print(torch.sum(logpygivenx))
    # print("logpxsum - logqxsum")
    # print(torch.sum(torch.logsumexp(logpxsum - logqxsum,0)))
    # print("logpz - logqzgivenx")
    # print(torch.sum(torch.logsumexp(logpz - logqzgivenx,0)))
    # # print("logprgivenxy")
    # # print(torch.sum(logprgivenxy))
    # # print("logqymgivenyor")
    # # print(torch.sum(logqymgivenyor))
    # print("---")
    
    return{'neg_bound':neg_bound, 'params_xm': params_xm, 'params_ym': params_ym, 'params_x': params_x,  'params_y': params_y, 'params_r':params_r, 'params_z':params_z, 'sum_logpr': sum_logpr,'sum_logpygivenx':sum_logpygivenx,'sum_logpx': sum_logpx,'sum_logqym': sum_logqym}
  #return{'neg_bound':neg_bound, 'params_xm': params_xm, 'params_ym': params_ym, 'params_y': params_y, 'params_r':params_r, 'sum_logpr': sum_logpr,'sum_logpygivenx': sum_logpygivenx,'sum_logpx': sum_logpx,'sum_logqym': sum_logqym}



  def impute(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, covar_miss, niws_z, temp):
    batch_size = iota_x.shape[0]
    tiled_iota_x = torch.Tensor.repeat(iota_x,[niws_z,1]).cuda(); tiled_iota_y = torch.Tensor.repeat(iota_y,[niws_z,1]).cuda()
    #tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[niws_z,1]).cuda(); tiled_tiled_iota_y = torch.Tensor.repeat(tiled_iota_y,[niws_z,1]).cuda()
    tiled_mask_x = torch.Tensor.repeat(mask_x,[niws_z,1]).cuda()#; tiled_tiled_mask_x = torch.Tensor.repeat(tiled_mask_x,[niws_z,1]).cuda()
    tiled_mask_y = torch.Tensor.repeat(mask_y,[niws_z,1]).cuda()#; tiled_tiled_mask_y = torch.Tensor.repeat(tiled_mask_y,[niws_z,1]).cuda()
    if not draw_miss: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[niws_z,1]).cuda(); tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[niws_z,1]).cuda()
    else: tiled_iota_xfull = None
    
    if covars: tiled_covars_miss = torch.Tensor.repeat(covar_miss,[M,1])
    else: tiled_covars_miss=None

    xincluded, yincluded, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx = forward(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, batch_size, niws_z, temp)
    
    if family=="Multinomial":
      yincluded=torch.nn.functional.one_hot(yincluded.to(torch.int64),num_classes=C).reshape([-1,C])
    
    # form of ELBO: log p(y|x) + log p(x) + log p(r|x) - log q(xm|xo, r)
    logpx_real = torch.Tensor([0]); logpx_cat = torch.Tensor([0]); logpx_count = torch.Tensor([0]); logpx_pos = torch.Tensor([0])
    logqxmgivenxor_real = torch.Tensor([0]); logqxmgivenxor_cat = torch.Tensor([0]); logqxmgivenxor_count = torch.Tensor([0]); logqxmgivenxor_pos = torch.Tensor([0])
    
    ## COMPUTE LOG PROBABILITIES ##
    if miss_x and miss_y:     # case 3
      # log p(r|x,y) # niw*niw*bs x p
      if not Ignorable:
        all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_mask_x[:,miss_ids], tiled_mask_y],1))
        logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M*M,batch_size])
      else: all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda()
      # log p(y|x)   # niw*niw*bs x p
      if family=="Gaussian": all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))    #yincluded: M*M*batch_size
      else: all_log_pygivenx = pygivenx.log_prob(yincluded)

      logpygivenx = all_log_pygivenx.reshape([niws_z*M*M,batch_size])
      
      ## log p(x) with VAE:
      if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M*M,batch_size])
      if exists_types[1]:
        logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M*M,batch_size])
        # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count])),axis=1).reshape([M*M,batch_size])
      if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M*M,batch_size])
      if exists_types[3]:
        # logpx_cat = torch.sum(p_x_cat.log_prob(xincluded[:,ids_cat]),axis=1).reshape([M*M,batch_size])
        for ii in range(0,p_cat):
          if ii==0: C0=0; C1=int(Cs[ii])
          else: C0=C1; C1=C0 + int(Cs[ii])
          
          if ii==0: logpx_cat = torch.sum(p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])   ),axis=1).reshape([-1,batch_size])
          else: logpx_cat = logpx_cat + torch.sum(p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])   ),axis=1).reshape([-1,batch_size])

      ## log q(xm|xo,r)
      # logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M*M,batch_size,p])).reshape([M*M*batch_size,p])*(1-tiled_mask_x),1).reshape([M*M,batch_size])
      if exists_types[0]: logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob( xincluded[:,ids_real].reshape([-1,p_real]) )*(1-tiled_mask_x[:,ids_real]),1).reshape([-1,batch_size])
      if exists_types[1]: logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([-1,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M*M,batch_size])
      if exists_types[2]: logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob( xincluded[:,ids_pos].reshape([-1,p_pos]) )*(1-tiled_mask_x[:,ids_pos]),1).reshape([-1,batch_size])
      if exists_types[3]:
        # logpx_cat = torch.sum(p_x_cat.log_prob(xincluded[:,ids_cat]),axis=1).reshape([M*M,batch_size])
        for ii in range(0,p_cat):
          if ii==0: C0=0; C1=int(Cs[ii])
          else: C0=C1; C1=C0 + int(Cs[ii])
          
          if ii==0: logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob(  xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])  )*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
          else: logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])   )*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

      logpxsum = torch.zeros([M*M,batch_size]).cuda(); logqxsum = torch.zeros([niws_z*M*M,batch_size]).cuda()
      
      # log q(ym|yo,r,xm,xo)
      logqymgivenyor = (qymgivenyor.log_prob(yincluded)*(1-tiled_mask_y)).reshape([niws_z*M*M,batch_size])
    else:
      # log p(r|x,y)
      if miss_x and not miss_y:  # case 1
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiled_mask_x[:,miss_ids])  # M*bs x p_miss
          logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M,batch_size])
        else: all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda()
        ## log q(xm|xo,r)
        # logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M,batch_size,-1])).reshape([M*batch_size,-1])*(1-tiled_mask_x),1).reshape([M,batch_size]); logqymgivenyor=0
        if exists_types[0]: logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob(xincluded[:,ids_real].reshape([niws_z*M*batch_size,p_real]))*(1-tiled_mask_x[:,ids_real]),1).reshape([niws_z*M,batch_size])
        if exists_types[1]:
          logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([niws_z*M*batch_size,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M,batch_size])
          # logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(torch.log(xincluded[:,ids_count]).reshape([M*M,batch_size,p_count])).reshape([M*M*batch_size,p_count])*(1-tiled_mask_x[:,ids_count]),1).reshape([M*M,batch_size])
        if exists_types[2]: logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob(xincluded[:,ids_pos].reshape([niws_z*M*batch_size,p_pos]))*(1-tiled_mask_x[:,ids_pos]),1).reshape([niws_z*M,batch_size])
        if exists_types[3]:
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([niws_z*M*batch_size,int(Cs[ii])]) )  *  (1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
            else: logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([niws_z*M*batch_size,int(Cs[ii])]) )  *  (1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

        # log p(y|x)   # all cases set to be (niw*niw*bs) x p
        # print(pygivenx.event_shape)
        # print(pygivenx.batch_shape)
        # print(yincluded.shape)
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1])) #yincluded: M*batch_size

        #print("Diagnostics:"); print(yincluded.shape); print(pygivenx.event_shape); print(pygivenx.batch_shape); print(all_log_pygivenx.shape)
        logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])
        
        ## log p(x): uncorrelated
        # logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        ## log p(x): correlated (unstructured)
        # just missing covars
        # logpx = p_x.log_prob(xincluded[:,miss_ids]).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        
        ## log p(x) with VAE:
        # print(xincluded[:,ids_real].shape)
        # print(torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real].reshape([niws_z*M,batch_size,p_real])),axis=2).shape)
        if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[1]:
          logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
          # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count]).reshape([M,batch_size,p_count])),axis=2).reshape([M,batch_size])
        if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[3]:
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logpx_cat = p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)] ).reshape([niws_z*M,batch_size])
            else: logpx_cat = logpx_cat + p_xs['cat'][ii].log_prob(  xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)]  ).reshape([niws_z*M,batch_size])
        

        logpxsum = torch.zeros([niws_z*M,batch_size]).cuda(); logqxsum = torch.zeros([niws_z*M,batch_size]).cuda()
      elif not miss_x and miss_y: # case 2
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiled_mask_y)
          logprgivenxy = all_logprgivenxy.reshape([niws_z*M,batch_size])   # no need to sum across columns (missingness of just y)
        else: all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda()
        # log q(ym|xo,r)
        logqymgivenyor = (qymgivenyor.log_prob(yincluded)*(1-tiled_mask_y)).reshape([niws_z*M,batch_size]); logqxsum=0
        # log p(y|x)   # all cases set to be (niw*niw*bs) x p
        
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])
        ## log p(x): uncorrelated
        # logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        ## log p(x): correlated (unstructured)

        # just missing covars
        # logpx = p_x.log_prob(xincluded[:,miss_ids]).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        
        ## log p(x) with VAE:
        if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[1]: 
          logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
          # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count])),axis=1).reshape([M,batch_size])
        if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
        if exists_types[3]: 
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logpx_cat = (p_xs['cat'].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C[ii]])   )).reshape([-1,batch_size])
            else: logpx_cat = logpx_cat + (p_xs['cat'].log_prob( xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C[ii]]) )).reshape([-1,batch_size])

        logpxsum = torch.zeros([M,batch_size]).cuda()
      else:     # no missing
        all_logprgivenxy=torch.Tensor([0]).cuda(); logprgivenxy=torch.Tensor([0]).cuda(); logqxsum=torch.Tensor([0]).cuda(); logqymgivenyor=torch.Tensor([0]).cuda()

        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([niws_z*1*batch_size])
        ## log p(x): uncorrelated
        # logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([1,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        ## log p(x): correlated (unstructured)

        ## just missing covars
        # logpx = p_x.log_prob(xincluded[:,miss_ids]).reshape([1,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
        
        ## log p(x) with VAE:
        if exists_types[0]: logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*1*batch_size])
        if exists_types[1]:
          logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*1*batch_size])
          # logpx_count = torch.sum(p_xs['count'].log_prob(torch.exp(xincluded[:,ids_count])),axis=1).reshape([1,batch_size])
        if exists_types[2]: logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*1*batch_size])
        if exists_types[3]:
          for ii in range(0,p_cat):
            if ii==0: C0=0; C1=int(Cs[ii])
            else: C0=C1; C1=C0 + int(Cs[ii])
            if ii==0: logpx_cat = (p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C])   )).reshape([-1])
            else: logpx_cat = logpx_cat + (p_xs['cat'].log_prob(   xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C])   )).reshape([-1])

        logpxsum = torch.zeros([niws_z*1,batch_size]).cuda(); logqxsum = torch.zeros([niws_z*1*batch_size]).cuda()

      
      ##logpx = torch.sum(p_x.log_prob(xincluded)*(1-tiled_mask_x),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (just missing x)
    
    if not Ignorable: sum_logpr = np.sum(logprgivenxy.cpu().data.numpy())
    else: sum_logpr = 0

    sum_logpygivenx = np.sum(logpygivenx.cpu().data.numpy())
    # sum_logpx = np.sum(logpx.cpu().data.numpy())
    sum_logpx = 0
    if exists_types[0]: sum_logpx = sum_logpx + np.sum(logpx_real.cpu().data.numpy())
    if exists_types[1]: sum_logpx = sum_logpx + np.sum(logpx_count.cpu().data.numpy())
    if exists_types[2]: sum_logpx = sum_logpx + np.sum(logpx_pos.cpu().data.numpy())
    if exists_types[3]: sum_logpx = sum_logpx + np.sum(logpx_cat.cpu().data.numpy())

    if miss_y: sum_logqym = np.sum(logqymgivenyor.cpu().data.numpy())
    else: sum_logqym = 0; logqymgivenyor=torch.Tensor([0]).cuda()

    #c = torch.cuda.memory_cached(0); a = torch.cuda.memory_allocated(0)
    #print("memory free:"); print(c-a)  # free inside cache


    # log q(ym|yo,r) ## add this into ELBO too
    # Case 1: x miss, y obs --> K samples of X
    # Case 2: x obs, y miss --> K samples of Y
    # Case 3: x miss, y miss --> K samples of X, K*M samples of Y. NEED TO MAKE THIS CONSISTENT: just make K=M and M samples of X and M samples of Y
    
    #initialized these sum terms earlier
    if exists_types[0]: logpxsum = logpxsum + logpx_real; logqxsum = logqxsum + logqxmgivenxor_real
    if exists_types[1]: logpxsum = logpxsum + logpx_count; logqxsum = logqxsum + logqxmgivenxor_count
    if exists_types[2]: logpxsum = logpxsum + logpx_pos; logqxsum = logqxsum + logqxmgivenxor_pos
    if exists_types[3]: logpxsum = logpxsum + logpx_cat; logqxsum = logqxsum + logqxmgivenxor_cat

    logpz = p_z.log_prob(zgivenx).reshape([-1,batch_size])
    # print(logpz.shape)
    # print(zgivenx.shape)
    # print(qzgivenx.event_shape)
    # print(qzgivenx.batch_shape)
    logqzgivenx = torch.sum(qzgivenx.log_prob(zgivenx.reshape([-1,batch_size,dim_z])),axis=2).reshape([-1,batch_size])
    
    #####################################################################################################
    ########################## NEED TO MAKE SURE THIS WORKS, IMPORTANCE WEIGHTED ################################
    #####################################################################################################
    
    # print("logpxsum:")
    # print(logpxsum.shape)
    # print("logpygivenx:")
    # print(logpygivenx.shape)
    
    IW = logpxsum + logpygivenx
    # print("IW:")
    # print(IW.shape)
    
    if not Ignorable: IW = IW + logprgivenxy
    # print("IW:")
    # print(IW.shape)
    
    if miss_x and miss_y:
      imp_weights = torch.nn.functional.softmax(IW + torch.Tensor.repeat(logpz,[M*M,1]) - torch.Tensor.repeat(logqzgivenx,[M*M,1]) - logqxsum - logqymgivenyor, 0)
    elif miss_x and not miss_y:
      imp_weights = torch.nn.functional.softmax(IW + torch.Tensor.repeat(logpz,[M,1]) - torch.Tensor.repeat(logqzgivenx,[M,1]) - logqxsum, 0)
    elif not miss_x and miss_y:
      imp_weights = torch.nn.functional.softmax(IW + torch.Tensor.repeat(logpz,[M,1]) - torch.Tensor.repeat(logqzgivenx,[M,1]) - logqymgivenyor, 0)
    else:
      imp_weights = torch.ones([1,batch_size])
    
    # print("xincluded:")
    # print(xincluded.shape)
    # print("imp_weights:")
    # print(imp_weights.shape)
    # print(yincluded.shape)
    xm = torch.einsum('ki,kij->ij', imp_weights.float(), xincluded.reshape([-1,batch_size,p]).float())
    xms = xincluded.reshape([-1,batch_size,p])
    # ym = torch.einsum('ki,kij->ij', imp_weights.float(), yincluded.reshape([-1,batch_size,1]).float())
    # yms = yincluded.reshape([-1,batch_size,1])
    if miss_y:
      ym = torch.einsum('ki,kij->ij', imp_weights.float(), yincluded.reshape([-1,batch_size,1]).float())
      yms = yincluded.reshape([-1,batch_size,1])
      # ym = torch.einsum('ki,kij->ij', imp_weights.float(), yincluded.reshape([-1,batch_size,C]).float())
      # yms = yincluded.reshape([-1,batch_size,C])
    else:
      ym = iota_y
      yms = iota_y
    
    ################################################################################
    ################################################################################
    ################################################################################
    # if ignorable:
    #   imp_weights = torch.nn.functional.softmax(torch.Tensor.repeat(logpxobsgivenz,[M,1]) + logpxmissgivenzsum.reshape([M*L,batch_size]) + torch.Tensor.repeat(logpz,[M,1,1]).reshape([M*L,batch_size]) - torch.Tensor.repeat(logqz,[M,1,1]).reshape([M*L,batch_size]) - logqxmissgivenzrsum.reshape([M*L,batch_size]),0) # these are w_1,....,w_L for all observations in the batch
    # else:
    #   imp_weights = torch.nn.functional.softmax(torch.Tensor.repeat(logpxobsgivenz,[M,1]) + logpxmissgivenzsum.reshape([M*L,batch_size]) + torch.Tensor.repeat(logpz,[M,1,1]).reshape([M*L,batch_size]) + logprgivenxy.reshape([M*L,batch_size]) - torch.Tensor.repeat(logqz,[M,1,1]).reshape([M*L,batch_size]) - logqxmissgivenzrsum.reshape([M*L,batch_size]),0) # these are w_1,....,w_L for all observations in the batch
    # 
    # xms = xincluded.reshape([M*L,batch_size,p])    # xincluded: [M*L*batch_size, p]
    # # print(xms)
    # xm = torch.einsum('ki,kij->ij', imp_weights.float(), xms.float())
    # ym = torch.einsum('ki,kij->ij', imp_weights.float(), yms.float())
    
    ################################################################################
    ################################################################################
    ################################################################################
    
    return {'xm': xm, 'ym': ym, 'imp_weights':imp_weights, 'xms': xms.detach(), 'yms': yms.detach()}
  # return {'xm': xm}
  
  # initialize weights
  # def weights_init(layer):
  #   if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
  
  # Define ADAM optimizer
  if (not Ignorable) and learn_r:   # is learn_r = Ignorable?
    params = list(NN_y.parameters()) + list(NN_r.parameters()) + list(encoder.parameters())# + list({'params': mu_x}) + list({'params': scale_x})
  else:
    params = list(NN_y.parameters()) + list(encoder.parameters())
  if miss_x:
    if exists_types[0]: params = params + list(NNs_xm['real'].parameters())#; NNs_xm['real'].apply(weights_init)
    if exists_types[1]: params = params + list(NNs_xm['count'].parameters())#; NNs_xm['count'].apply(weights_init)
    if exists_types[2]: params = params + list(NNs_xm['pos'].parameters())#; NNs_xm['pos'].apply(weights_init)
    if exists_types[3]:
      for ii in range(0, p_cat):
        params = params + list(NNs_xm['cat'][ii].parameters())#; NNs_xm['cat'][ii].apply(weights_init)
  if miss_y: params = params + list(NN_ym.parameters())
  
  #encoder.apply(weights_init)
  if exists_types[0]: params = params + list(decoders['real'].parameters())#; decoders['real'].apply(weights_init)
  if exists_types[1]: params = params + list(decoders['count'].parameters())#; decoders['count'].apply(weights_init)
  if exists_types[2]: params = params + list(decoders['pos'].parameters())#; decoders['pos'].apply(weights_init)
  if exists_types[3]:
    for ii in range(0,p_cat):
      params = params + list(decoders['cat'][ii].parameters())#; decoders['cat'][ii].apply(weights_init)
  
  
  optimizer = optim.Adam(params,lr=lr)
  # optimizer.add_param_group({"params":mu_x})
  # optimizer.add_param_group({"params":scale_x})
  optimizer.add_param_group({"params":alpha})

  # Train and impute every 100 epochs
  mse_train_miss_x=np.array([])
  mse_train_obs_x=np.array([])
  mse_train_miss_y=np.array([])
  mse_train_obs_y=np.array([])
  #mse_pr_epoch = np.array([])
  #CEL_epoch=np.array([]) # Cross-entropy error
  xhat = np.copy(xhat_0) # This will be out imputed data matrix
  yhat = np.copy(yhat_0) # This will be out imputed data matrix

  #trace_ids = np.concatenate([np.where(R[:,0]==0)[0][0:2],np.where(R[:,0]==1)[0][0:2]])
  trace_ids = np.arange(0,10)
  if (trace): print(xhat_0[trace_ids])

  # if miss_x: NN_xm.apply(weights_init)
  # if miss_y: NN_ym.apply(weights_init)
  # NN_y.apply(weights_init)
  # if (learn_r and not Ignorable): NN_r.apply(weights_init)
  
  time_train=[]
  time_impute=[]
  LB_epoch=[]
  val_LB_epoch=[]
  sum_logpy_epoch =[]
  sum_logqym_epoch=[]
  sum_logpr_epoch=[]
  sum_logpx_epoch=[]
  
  early_stopped = False  # will be changed to True if early stop happens
  early_stop_epochs = n_epochs
  early_stop_check_epochs = 10       # relative change in val_LB checked across this many epochs
  early_stop_tol = 0.0001               # tolerance of change in val_LB across early_stop_check_epochs
  patience_index = 0; patience = 25
  opt_model = {}   # initialize dictionary to save opt_model
  opt_val_LB = -sys.float_info.max   # largest negative value float --> to be replaced in first check
  
  # only assign xfull to cuda if it's necessary (save GPU ram)
  if not draw_miss: cuda_xfull = torch.from_numpy(xfull).float().cuda(); cuda_yfull = torch.from_numpy(yfull).float().cuda()
  else: cuda_xfull = None; cuda_yfull = None
  
  ## Initialize params_
  if family=="Gaussian":
    params_y = {'mean': np.empty([n,1]) , 'scale': np.empty([n,1])}          # N x 1
    params_ym = {'mean': np.empty([n,1]) , 'scale': np.empty([n,1])}
  elif family=="Poisson":
    # log transformed --> cts?
    params_y = {'lambda': np.empty([n,1])}
    params_ym = {'mean': np.empty([n,1]), 'scale': np.empty([n,1])}  ### PROBLEM. are we going to transform response that is count too??
  elif family=="Multinomial":
    params_y = {'probs': np.empty([n,C])}
    params_ym = {'probs': np.empty([n,C])}                   # N x C
  params_x={}; params_x['cat']=[]
  params_xm={}; params_xm['cat']=[]
  if exists_types[0]:
    params_x['real'] = {'mean': np.empty([n,p_real]), 'scale': np.empty([n,p_real])}
    if miss_x: params_xm['real'] = {'mean': np.empty([n,p_real]), 'scale': np.empty([n,p_real])}      # N x p_real
  if exists_types[1]:
    params_x['count'] = {'mean': np.empty([n,p_count]), 'scale': np.empty([n,p_count])}
    # params_x['count'] = {'lambda': np.empty([n,p_count])}
    if miss_x: params_xm['count'] = {'mean': np.empty([n,p_count]), 'scale': np.empty([n,p_count])}    # N x p_count
  if exists_types[2]:
    params_x['pos'] = {'mean': np.empty([n,p_pos]), 'scale': np.empty([n,p_pos])}
    if miss_x: params_xm['pos'] = {'mean': np.empty([n,p_pos]), 'scale': np.empty([n,p_pos])}      # N x p_real
  if exists_types[3]:
    for ii in range(0, p_cat):
      params_x['cat'].append(np.empty([n,int(Cs[ii])]))                           # put matrix of N x C here. prob torch.zeros()
      if miss_x: params_xm['cat'].append(np.empty([n,int(Cs[ii])]))
  params_r = {'probs': np.empty([n,n_params_r])}                                    # N x n_params_r
  params_z = {'mean': np.empty([n,dim_z]), 'scale': np.empty([n,dim_z])}
  
  sys.stdout.flush()
  
  if train==1:
    print("Training")
    # Training+Imputing
    for ep in range(1,n_epochs):
      # if ep == 10: sys.stdout.flush(); raise NameError('Stopped at 10 epochs')
      
      if ep % 10==0:
        print("Epoch " + str(ep))
      sys.stdout.flush()
      # print("Epoch" + str(ep))
      # t = torch.cuda.get_device_properties(0).total_memory
      # r = torch.cuda.memory_reserved(0) 
      # a = torch.cuda.memory_allocated(0)
      # f = r-a  # free inside reserved
      # print("before epoch " + str(ep) + " free memory:", str(f))
      
      if unbalanced:
        # for Unbalanced Y class variable
        # Create custom splits to draw same # of obs from each class
        ids_classes_y = npi.group_by(Y).split(range(0,n))   # length-K array of indices of values for each category of Y
        n_classes_y = [len(i) for i in ids_classes_y]
        
        n_majority, n_classes_minority = np.max(n_classes_y), np.min(n_classes_y)
        bs2 = min([n_classes_minority, np.floor(bs/len(ids_classes_y))])     # number drawn from each class: should be at least the total of smallest class, and at most the original bs/#classes
        ids_majority = ids_classes_y[np.argmax(n_classes_y)]
        np.random.shuffle( ids_majority )
        
        n_partitions = np.ceil(n_majority/bs2)    # number of partitions (times to update in epoch)
        splits = np.array_split(ids_majority, n_partitions)    # split the majority class equally first (append samples of nonmajority later)
        n_splits_majority = [len(i) for i in splits]    # number in each split in majority class --> should be less than bs in each split
        
        ids_nonmajority_classes_y = ids_classes_y        # create first, then delete majority class. Sample from these IDs equally
        ids_nonmajority_classes_y.pop(np.argmax(n_classes_y))
        
        
        ## Nested for loop to draw the same number of samples (as in each majority class split) from minority classes, without replacement.
        ## may be a more efficient way to do this
        for i in range(0,len(splits)):
          for j in range(0,len(ids_nonmajority_classes_y)):
            splits[i] = np.append(splits[i], np.random.choice(ids_nonmajority_classes_y[j], n_splits_majority[i], replace=False))
        # [len(i) for i in splits]    # should be less than bs in each split (accomplished by "ceil")
        # [len(np.unique(i)) != len(i) for i in splits]    # no duplicates
        
        batches_xfull = [xfull[i,:] for i in splits]
        batches_x = [xhat_0[i,:] for i in splits]
        batches_yfull = [yfull[i] for i in splits]
        batches_y = [yhat_0[i] for i in splits]
        batches_mask_x = [mask_x[i,:] for i in splits]
        batches_mask_y = [mask_y[i,:] for i in splits]
        if covars: batches_covar = [covars_miss[i] for i in splits]
      else:
        perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
        batches_xfull = np.array_split(xfull[perm,],n/bs)
        batches_x = np.array_split(xhat_0[perm,], n/bs)
        batches_yfull = np.array_split(yfull[perm,],n/bs)
        batches_y = np.array_split(yhat_0[perm,], n/bs)
        batches_mask_x = np.array_split(mask_x[perm,], n/bs)     # only mask for x. for y --> include as a new mask
        batches_mask_y = np.array_split(mask_y[perm,], n/bs)     # only mask for x. for y --> include as a new mask
        if covars: batches_covar = np.array_split(covars_miss[perm,], n/bs)
        splits = np.array_split(perm,n/bs)
      batches_loss = []
      t0_train=time.time()
      for it in range(len(batches_x)):
      # for it in range(8):        # testing 10% minibatches per epoch, with low minibatch size
        if (not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
        else: b_xfull = None; b_yfull = None
        b_x = torch.from_numpy(batches_x[it]).float().cuda()
        b_y = torch.from_numpy(batches_y[it]).float().cuda()
        b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
        b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None
        
        optimizer.zero_grad()
        if miss_x: 
          if exists_types[0]: NNs_xm['real'].zero_grad()
          if exists_types[1]: NNs_xm['count'].zero_grad()
          if exists_types[2]: NNs_xm['pos'].zero_grad()
          if exists_types[3]:
            for ii in range(0,p_cat):
              NNs_xm['cat'][ii].zero_grad()

        if miss_y: NN_ym.zero_grad()
        NN_y.zero_grad()
        
        if (learn_r and not Ignorable): NN_r.zero_grad()
        encoder.zero_grad()
        if exists_types[0]: decoders['real'].zero_grad()
        if exists_types[1]: decoders['count'].zero_grad()
        if exists_types[2]: decoders['pos'].zero_grad()
        if exists_types[3]:
          for ii in range(0,p_cat):
            decoders['cat'][ii].zero_grad()
        #mu_x.zero_grad(); scale_x.zero_grad()

        loss_fit = compute_loss(iota_xfull=b_xfull, iota_yfull=b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, temp=temp)
        temp_params_x, temp_params_xm, temp_params_ym, temp_params_y, temp_params_r, temp_params_z = loss_fit['params_x'], loss_fit['params_xm'], loss_fit['params_ym'], loss_fit['params_y'], loss_fit['params_r'], loss_fit['params_z']

        ## inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,temp
        ######################################################################################################
        ##################### need to output parameters of each distribution, batched ########################
        ##################### un-comment out "all_params". NEED all_params$y for prediction ##################
        ######################################################################################################
        
        # loss = loss_fit['neg_bound']
        loss = loss_fit['neg_bound']/(niws_z*M*n)
        
        ############### L1 weight regularization #############
        if learn_r and not Ignorable:
          L1_reg = torch.tensor(0., requires_grad=True).cuda()
          for name, param in NN_r[0].named_parameters():
            if 'weight' in name:
              L1_reg = L1_reg + torch.norm(param, 1)
          loss = loss + L1_weight*L1_reg
        ######################################################
        
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        
        if exists_types[0]:
          params_x['real']['mean'][splits[it],:] = temp_params_x['real']['mean']; params_x['real']['scale'][splits[it],:] = temp_params_x['real']['scale']
          if miss_x: params_xm['real']['mean'][splits[it],:] = temp_params_xm['real']['mean']; params_xm['real']['scale'][splits[it],:] = temp_params_xm['real']['scale']
        if exists_types[1]:
          params_x['count']['mean'][splits[it],:] = temp_params_x['count']['mean']; params_x['count']['scale'][splits[it],:] = temp_params_x['count']['scale']
          # params_x['count']['lambda'][splits[it],:] = temp_params_x['count']['lambda']
          if miss_x: params_xm['count']['mean'][splits[it],:] = temp_params_xm['count']['mean']; params_xm['count']['scale'][splits[it],:] = temp_params_xm['count']['scale']
        if exists_types[2]:
          params_x['pos']['mean'][splits[it],:] = temp_params_x['pos']['mean']; params_x['pos']['scale'][splits[it],:] = temp_params_x['pos']['scale']
          if miss_x: params_xm['pos']['mean'][splits[it],:] = temp_params_xm['pos']['mean']; params_xm['pos']['scale'][splits[it],:] = temp_params_xm['pos']['scale']
        if exists_types[3]:
          for ii in range(0, p_cat):
            params_x['cat'][ii][splits[it],:] = temp_params_x['cat'][ii]
            if miss_x: params_xm['cat'][ii][splits[it],:] = temp_params_xm['cat'][ii]
        
        if family=="Gaussian":
          params_y['mean'][splits[it],:] = temp_params_y['mean']; params_y['scale'][splits[it],:] = temp_params_y['scale']
          if miss_y: params_ym['mean'][splits[it],:] = temp_params_ym['mean']; params_ym['scale'][splits[it],:] = temp_params_ym['scale']
        elif family=="Poisson":
          params_y['lambda'][splits[it],:] = temp_params_y['lambda']
          if miss_y: params_ym['mean'][splits[it],:] = temp_params_ym['mean']; params_ym['scale'][splits[it],:] = temp_params_ym['scale']
        elif family=="Multinomial":
          params_y['probs'][splits[it],:] = temp_params_y['probs']
          if miss_y: params_ym['probs'][splits[it],:] = temp_params_ym['probs']

        if not Ignorable: params_r['probs'][splits[it],:] = temp_params_r['probs']
        params_z['mean'][splits[it],:] = temp_params_z['mean']; params_z['scale'][splits[it],:] = temp_params_z['scale']
        # torch.autograd.set_detect_anomaly(True)
        
        # if torch.isnan(loss):
        #   print("px real mean and scale:")
        #   print(params_x['real']['mean'][splits[it],:][:20])
        #   print(params_x['real']['scale'][splits[it],:][:20])
        #   print("px count mean and scale:")
        #   print(params_x['count']['mean'][splits[it],:][:20])
        #   print(params_x['count']['scale'][splits[it],:][:20])
        #   print("qxm count mean and scale:")
        #   print(params_xm['count']['mean'][splits[it],:][:20])
        #   print(params_xm['count']['scale'][splits[it],:][:20])
        #   sys.exit("NA loss. Printing loss_fit object to debug")
        
        loss.backward()
        optimizer.step()
        
        # Impose L1 thresholding to 0 for weight if norm < 1e-2
        if learn_r and not Ignorable and L1_weight>0: #or L2_weight>0:
          with torch.no_grad(): NN_r[0].weight[torch.abs(NN_r[0].weight) < L1_weight] = 0           ####################### NEW
        # print("params_x (first 2)")
        # # print(params_x)
        # print("real: mean and scale")
        # print(params_x['real']['mean'][splits[it],:][:20])
        # print(params_x['real']['scale'][splits[it],:][:20])
        # print("count: lambda")
        # print(params_x['count']['lambda'][splits[it],:][:20])
        # print("params_xm (first 2)")
        # # print(params_xm)
        # print("real: mean and scale")
        # print(params_xm['real']['mean'][splits[it],:][:20])
        # print(params_xm['real']['scale'][splits[it],:][:20])
        # print("count: mean and scale")
        # print(params_xm['count']['mean'][splits[it],:][:20])
        # print(params_xm['count']['scale'][splits[it],:][:20])
      time_train=np.append(time_train,time.time()-t0_train)
      
      if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      else: torch_covars_miss = None

      
      total_loss = np.sum(batches_loss)
      
      if arch=="VAE":
        # LB = -total_loss/(M**(int(miss_x) + int(miss_y))*n)
        # LB = -total_loss/(niws_z*M*n)
        LB = -total_loss
      elif arch=="IWAE":
        # LB = -total_loss/(M**(int(miss_x) + int(miss_y))*n) + (int(miss_x) + int(miss_y)) * np.log(M) # miss_x^2 + miss_y^2 = 1 if one miss, 2 if both miss     # redundant: + np.log(M)*(miss_x and miss_y)
        # LB = -total_loss/(niws_z*M*n) + np.log(niws_z) + np.log(M) # miss_x^2 + miss_y^2 = 1 if one miss, 2 if both miss     # redundant: + np.log(M)*(miss_x and miss_y)
        LB = -total_loss + np.log(niws_z) + np.log(M) # miss_x^2 + miss_y^2 = 1 if one miss, 2 if both miss     # redundant: + np.log(M)*(miss_x and miss_y)
      
      # print("Epoch " + str(ep) + ", LB = " + str(LB))
      LB_epoch=np.append(LB_epoch,LB)
      
      ### Add early stop criterion here ######
      #### need as extra input: early_stop, X_val, Rx_val, Y_val, Ry_val
      if early_stop:
        n_val = xfull_val.shape[0]
        bs_val = min(bs, n_val)
        ##################################################################
        ###### COMPUTE VALIDATION LOSS (for early stopping criteria) #####
        ##################################################################
        
        # if unbalanced:   ##### commented out for now: validation may not require different treatment of unbalanced. (would this even make sense when validating?)
        #   # for Unbalanced Y class variable
        #   # Create custom splits to draw same # of obs from each class
        #   ids_classes_y = npi.group_by(Y_val).split(range(0,n_val))   # length-K array of indices of values for each category of Y
        #   n_classes_y = [len(i) for i in ids_classes_y]
        #   
        #   n_majority, n_classes_minority = np.max(n_classes_y), np.min(n_classes_y)
        #   bs_val2 = min([n_classes_minority, np.floor(bs_val/len(ids_classes_y))])     # number drawn from each class: should be at least the total of smallest class, and at most the original bs/#classes
        #   ids_majority = ids_classes_y[np.argmax(n_classes_y)]
        #   np.random.shuffle( ids_majority )
        #   
        #   n_partitions = np.ceil(n_majority/bs2)    # number of partitions (times to update in epoch)
        #   splits = np.array_split(ids_majority, n_partitions)    # split the majority class equally first (append samples of nonmajority later)
        #   n_splits_majority = [len(i) for i in splits]    # number in each split in majority class --> should be less than bs in each split
        #   
        #   ids_nonmajority_classes_y = ids_classes_y        # create first, then delete majority class. Sample from these IDs equally
        #   ids_nonmajority_classes_y.pop(np.argmax(n_classes_y))
        #   
        #   
        #   ## Nested for loop to draw the same number of samples (as in each majority class split) from minority classes, without replacement.
        #   ## may be a more efficient way to do this
        #   for i in range(0,len(splits)):
        #     for j in range(0,len(ids_nonmajority_classes_y)):
        #       splits[i] = np.append(splits[i], np.random.choice(ids_nonmajority_classes_y[j], n_splits_majority[i], replace=False))
        #   # [len(i) for i in splits]    # should be less than bs in each split (accomplished by "ceil")
        #   # [len(np.unique(i)) != len(i) for i in splits]    # no duplicates
        #   
        #   batches_xfull = [xfull_val[i,:] for i in splits]
        #   batches_x = [xhat_0_val[i,:] for i in splits]
        #   batches_yfull = [yfull_val[i] for i in splits]
        #   batches_y = [yhat_0_val[i] for i in splits]
        #   batches_mask_x = [mask_x_val[i,:] for i in splits]
        #   batches_mask_y = [mask_y_val[i,:] for i in splits]
        #   if covars: batches_covar = [covars_miss[i] for i in splits]
        # else:
        perm = np.random.permutation(n_val) # We use the "random reshuffling" version of SGD
        if (not draw_miss) and (not Ignorable): batches_xfull = np.array_split(xfull_val[perm,],n_val/bs_val); batches_yfull = np.array_split(yfull_val[perm,],n_val/bs_val)
        batches_x = np.array_split(xhat_0_val[perm,], n_val/bs_val)
        batches_y = np.array_split(yhat_0_val[perm,], n_val/bs_val)
        batches_mask_x = np.array_split(mask_x_val[perm,], n_val/bs_val)
        batches_mask_y = np.array_split(mask_y_val[perm,], n_val/bs_val)
        if covars: batches_covar = np.array_split(covars_miss[perm,], n_val/bs_val)
        #batches_prM = np.array_split(prM[perm,],n/bs)
        splits = np.array_split(perm,n_val/bs_val)
        
        # minibatch save:
        # losses
        batches_val_loss = []
        for it in range(len(batches_x)):
          
          if (not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
          else: b_xfull = None; b_yfull = None
          b_x = torch.from_numpy(batches_x[it]).float().cuda()
          b_y = torch.from_numpy(batches_y[it]).float().cuda()
          b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
          b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
          if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
          else: b_covar = None
          
          optimizer.zero_grad()
          if miss_x: 
            if exists_types[0]: NNs_xm['real'].zero_grad()
            if exists_types[1]: NNs_xm['count'].zero_grad()
            if exists_types[2]: NNs_xm['pos'].zero_grad()
            if exists_types[3]:
              for ii in range(0,p_cat):
                NNs_xm['cat'][ii].zero_grad()
          if miss_y: NN_ym.zero_grad()
          NN_y.zero_grad()
          if (learn_r and not Ignorable): NN_r.zero_grad()
          encoder.zero_grad()
          if exists_types[0]: decoders['real'].zero_grad()
          if exists_types[1]: decoders['count'].zero_grad()
          if exists_types[2]: decoders['pos'].zero_grad()
          if exists_types[3]:
            for ii in range(0,p_cat):
              decoders['cat'][ii].zero_grad()
          #mu_x.zero_grad(); scale_x.zero_grad()
  
          loss_fit = compute_loss(iota_xfull=b_xfull, iota_yfull=b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, temp=temp)

          loss = loss_fit['neg_bound']
          batches_val_loss = np.append(batches_val_loss, loss.cpu().data.numpy())
          
          loss_fit.pop("neg_bound")
          
          
        # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,temp
        temp_params_x, temp_params_xm, temp_params_ym, temp_params_y, temp_params_r, temp_params_z = loss_fit['params_x'], loss_fit['params_xm'], loss_fit['params_ym'], loss_fit['params_y'], loss_fit['params_r'], loss_fit['params_z']

        
        total_val_loss = np.sum(batches_val_loss)
        
        if arch=="VAE":
          # LB = -total_loss/(M**(int(miss_x) + int(miss_y))*n)
          val_LB = -total_val_loss/(niws_z*M*n_val)
        elif arch=="IWAE":
          # LB = -total_loss/(M**(int(miss_x) + int(miss_y))*n) + (int(miss_x) + int(miss_y)) * np.log(M) # miss_x^2 + miss_y^2 = 1 if one miss, 2 if both miss     # redundant: + np.log(M)*(miss_x and miss_y)
          val_LB = -total_val_loss/(niws_z*M*n_val) + np.log(niws_z) + np.log(M) # just 1 now: sampling M times no matter miss_x, miss_y or not either     # redundant: + np.log(M)*(miss_x and miss_y)
        val_LB_epoch=np.append(val_LB_epoch,val_LB)
        
        # print("validation LB: " + str(val_LB))
        
        # Start checking for early_stop after 10 epochs and when temperature hits minimum
        if (ep > early_stop_check_epochs) and temp.item() <= temp_min.item():
          ## stopping right away if relative change in LB < early_stop_tol
          # rel_delta_val_LB = (val_LB_epoch[ep] - val_LB_epoch[(ep) - early_stop_check_epochs])/np.absolute(val_LB_epoch[(ep) - early_stop_check_epochs] + 1e-50)  # 1e-50 for stability in denom
          # if rel_delta_val_LB < early_stop_tol:
          #   early_stopped = True
          #   print('Early stopping at epoch %d!' %ep)
          #   early_stop_epochs = ep
          if learn_r and (not Ignorable): saved_model={'NNs_xm': NNs_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r,'encoder':encoder, 'decoders': decoders}
          else: saved_model={'NNs_xm': NNs_xm, 'NN_ym': NN_ym, 'NN_y': NN_y,'encoder':encoder, 'decoders': decoders}
          
          ## stop after patience = 50 epochs. save best model from these
          # print("ep" + str(ep))
          # print("val_LB_epoch length" + str(len(val_LB_epoch)) )
          # if val_LB_epoch[ep-1] > val_LB_epoch[ep-2]:
          # print("Diagnostics... ep = " + str(ep) + ", early_stop_check_epochs = " + str(early_stop_check_epochs))
          # print("val_LB_epoch[ep-1]: " + str(val_LB_epoch[ep-1]) + ", opt_val_LB: " + str(opt_val_LB) + ", early_stop_tol: " + str(early_stop_tol))
          if (ep == early_stop_check_epochs + 1) or (val_LB_epoch[ep-1] > opt_val_LB + abs(early_stop_tol*opt_val_LB) ):     # if the increase is greater than opt_val_LB by 0.01%
            opt_LB = LB
            opt_val_LB = val_LB_epoch[ep-1]#; opt_model = saved_model
            torch.save(saved_model, dir_name + "/temp_model.pth")
            opt_params = {'x': params_x, 'xm': params_xm, 'y': params_y, 'ym': params_ym, 'r': params_r, 'z': params_z}
            patience_index = 0 # reset patience to 0
            # print("opt_val_LB: " + str(opt_val_LB))
          else:
            if val_LB_epoch[ep-1] > opt_val_LB:   # if val_LB is larger the opt_val_LB by any value, still replace the saved model, but don't reset patience unless it is greater by a certain amt
              opt_LB = LB
              torch.save(saved_model, dir_name + "/temp_model.pth")
              opt_params = {'x': params_x, 'xm': params_xm, 'y': params_y, 'ym': params_ym, 'r': params_r, 'z': params_z}
            patience_index = patience_index + 1
            # print("patience: " + str(patience_index))
            if patience_index >= patience:
              early_stopped = True
              early_stop_epochs = ep
              print(val_LB_epoch[-(min(int(patience),ep-1)):])
              print("Early stop criterion met. Reverting to optimal model, and imputing one last time and ending training")
              saved_model = torch.load(dir_name + "/temp_model.pth")
              os.remove(dir_name + "/temp_model.pth")
              NNs_xm = saved_model['NNs_xm']; NN_ym = saved_model['NN_ym']; NN_y = saved_model['NN_y']
              encoder = saved_model['encoder']; decoders = saved_model['decoders']
              if (learn_r and not Ignorable): NN_r = saved_model['NN_r']
      else: val_LB=None
      
      if ep % 100 == 1 or early_stopped:
        print('Epoch %g' %ep)
        print('Likelihood lower bound  %g' %LB) # Gradient step
        print("temp: " + str(temp))

        if trace:
          print("no loss fit on full data")
          # if family=="Gaussian":
          #   print("mean, p(y|x):")    # E[y|x] = beta0 + beta*x
          #   print(torch.mean(loss_fit['params_y']['mean'].reshape([M,-1]),0).reshape([-1,1])[trace_ids])
          #   print("scale, p(y|x):")
          #   print(torch.mean(loss_fit['params_y']['scale'].reshape([M,-1]),0).reshape([-1,1])[trace_ids])
          # elif family=="Multinomial":
          #   print("probs, p(y|x):")
          #   print(torch.mean(loss_fit['params_y']['probs'].reshape([M,-1]),0).reshape([-1,C])[trace_ids])
          # elif family=="Poisson":
          #   print("lambda, p(y|x):")
          #   print(torch.mean(loss_fit['params_y']['lambda'].reshape([M,-1]),0).reshape([-1,1])[trace_ids])
          # 
          # if miss_x:
          #   print("mean (avg over M samples), q(xm|xo,r):")
          #   print(loss_fit['params_xm']['mean'][trace_ids])
          #   print("scale (avg over M samples), q(xm|xo,r):")
          #   print(loss_fit['params_xm']['scale'][trace_ids])
          # if miss_y:
          #   print("mean (avg over M samples), q(ym|yo,r,xm,xo):")
          #   print(loss_fit['params_ym']['mean'][trace_ids])
          #   print("scale (avg over M samples), q(ym|yo,r,xm,xo):")
          #   print(loss_fit['params_ym']['scale'][trace_ids])
          # 
          # if not Ignorable:
          #   print("prob_Missing (avg over M, then K samples):")
          #   print(torch.mean(loss_fit['params_r']['probs'].reshape([M,-1]),axis=0).reshape([n,-1])[trace_ids])
        
        ## For imputation, unbalanced response doesn't matter.
        t0_impute=time.time()
        batches_xfull = np.array_split(xfull,n/impute_bs)
        batches_yfull = np.array_split(yfull,n/impute_bs)
        batches_x = np.array_split(xhat_0, n/impute_bs)
        batches_y = np.array_split(yhat_0, n/impute_bs)
        batches_mask_x = np.array_split(mask_x, n/impute_bs)
        batches_mask_y = np.array_split(mask_y, n/impute_bs)
        if covars: batches_covar = np.array_split(covars_miss, n/impute_bs)
        splits = np.array_split(range(n),n/impute_bs)
        for it in range(len(batches_x)):
          if (not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
          else: b_xfull = None; b_yfull=None
          b_x = torch.from_numpy(batches_x[it]).float().cuda()
          b_y = torch.from_numpy(batches_y[it]).float().cuda()
          b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
          b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
          if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
          else: b_covar = None
          impute_fit=impute(iota_xfull = b_xfull, iota_yfull = b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, niws_z=niws_z, temp=temp)
          # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,L,temp

          # imputing xmiss:
          b_xhat = xhat[splits[it],:]
          b_yhat = yhat[splits[it],:]
          #b_xhat[batches_mask_x[it]] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:].cpu().data.numpy()[batches_mask_x[it]] # observed data. nop need to impute
          if miss_x: b_xhat[~batches_mask_x[it]] = impute_fit['xm'].cpu().data.numpy()[~batches_mask_x[it]]       # just missing impute
          if miss_y: b_yhat[~batches_mask_y[it]] = impute_fit['ym'].cpu().data.numpy()[~batches_mask_y[it]]       # just missing impute
          xhat[splits[it],:] = b_xhat
          yhat[splits[it],:] = b_yhat

        time_impute=np.append(time_impute,time.time()-t0_impute)
        
        # err_x = mse(xhat,xfull,mask_x)
        print("xfull (first 5):")
        print(xfull[:min(5,n), :min(5,p)])
        print("xhat (first 5):")
        print(xhat[:min(5,n), :min(5,p)])
        print("mask_x (first 5):")
        print(mask_x[:min(5,n), :min(5,p)])
        # if exists_types[0]: err_x_real = mse(((xhat*norm_sds_x) + norm_means_x)[ids_real], ((xfull*norm_sds_x)+norm_means_x)[ids_real], mask_x[:,ids_real])
        # if exists_types[1]: err_x_count = mse(((xhat*norm_sds_x) + norm_means_x)[ids_count], ((xfull*norm_sds_x)+norm_means_x)[ids_count], mask_x[:,ids_count])
        if exists_types[0]: err_x_real = mse(xhat[:,ids_real], xfull[:,ids_real], mask_x[:,ids_real])
        if exists_types[1]: err_x_count = mse(xhat[:,ids_count], xfull[:,ids_count], mask_x[:,ids_count])
        if exists_types[2]: err_x_pos = mse(xhat[:,ids_pos], xfull[:,ids_pos], mask_x[:,ids_pos])
        if exists_types[3]:
          err_x_cat0 = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat]) #err_x_cat = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat])
          if np.sum(np.isnan(xfull[:,ids_cat])) == 0:
            err_x_cat1 = pred_acc(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat],Cs) #err_x_cat = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat])
          else:
            err_x_cat1 = {'miss': np.nan, 'obs': np.nan}

        if family=="Multinomial": err_y = pred_acc(yhat, yfull, mask_y, C)
        else: err_y = mse(yhat*norm_sd_y+norm_mean_y, yfull*norm_sd_y+norm_mean_y, mask_y)

        # mse_train_miss_x = np.append(mse_train_miss_x,np.array([err_x['miss']]),axis=0)
        # mse_train_obs_x = np.append(mse_train_obs_x,np.array([err_x['obs']]),axis=0)
        # mse_train_miss_y = np.append(mse_train_miss_y,np.array([err_y['miss']]),axis=0)
        # mse_train_obs_y = np.append(mse_train_obs_y,np.array([err_y['obs']]),axis=0)
        
        if exists_types[0]:
          print('Observed MSE x_real:  %g' %err_x_real['obs'])   # these aren't reconstructed/imputed
          print('Missing MSE x_real:  %g' %err_x_real['miss'])
          print("L1 (missing):")
          print(np.mean(np.abs(((xhat-xfull)*norm_sds_x)[:,ids_real])[~mask_x[:,ids_real]]))
          print("L1 (observed):")
          print(np.mean(np.abs(((xhat-xfull)*norm_sds_x)[:,ids_real])[mask_x[:,ids_real]]))
        if exists_types[1]:
          print('Observed MSE x_count (log-transformed):  %g' %err_x_count['obs'])   # these aren't reconstructed/imputed
          print('Missing MSE x_count (log-transformed):  %g' %err_x_count['miss'])
          print("L1 (missing):")
          print(np.mean(np.abs(((xhat-xfull)*norm_sds_x)[:,ids_count])[~mask_x[:,ids_count]]))
          print("L1 (observed):")
          print(np.mean(np.abs(((xhat-xfull)*norm_sds_x)[:,ids_count])[mask_x[:,ids_count]]))
        if exists_types[2]:
          print('Observed MSE x_pos:  %g' %err_x_pos['obs'])   # these aren't reconstructed/imputed
          print('Missing MSE x_pos:  %g' %err_x_pos['miss'])
          print("L1 (missing):")
          print(np.mean(np.abs(((xhat-xfull)*norm_sds_x)[:,ids_pos])[~mask_x[:,ids_pos]]))
          print("L1 (observed):")
          print(np.mean(np.abs(((xhat-xfull)*norm_sds_x)[:,ids_pos])[mask_x[:,ids_pos]]))
        if exists_types[3]:
          print('Observed MSE x_cat:  %g' %err_x_cat0['obs'])   # these aren't reconstructed/imputed
          print('Missing MSE x_cat:  %g' %err_x_cat0['miss'])
          print('Observed Pred_Acc x_cat:  %g' %err_x_cat1['obs'])   # these aren't reconstructed/imputed
          print('Missing Pred_Acc x_cat:  %g' %err_x_cat1['miss'])
        print('Observed MSE/Pred_Acc y (Gaussian/Multinomial):  %g' %err_y['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE/Pred_Acc y (Gaussian/Multinomial):  %g' %err_y['miss'])
        print("NN_y bias:")
        print((NN_y[0].bias).cpu().data.numpy())
        print("NN_y weights (first 5):")
        print(np.around(((NN_y[0].weight).cpu().data.numpy())[:5,:5], decimals=3))
        print("Y (first 5):")
        print(yhat_0[:5])
        print("params_y (first 5):")
        if family=="Gaussian":
          print("params_y mean (first 5)")
          print(params_y['mean'][:5])
        elif family=="Multinomial":
          print("params_y probs (first 5)")
          print(np.around(params_y['probs'][:5], decimals = 3))
        elif family=="Poisson":
          print("params_y lambda (first 5)")
          print(params_y['lambda'][:5])
        if not Ignorable:
          print("NN_r bias (first 5):")
          print(np.around((NN_r[0].bias[:5]).cpu().data.numpy(), decimals=3))
          print("NN_r weights (" + str(pr) + " cols = input, " + str(n_params_r) + " rows = output) (first 4):")
          print(np.around((NN_r[0].weight[0:min(4,n_params_r),0:min(4,pr)]).cpu().data.numpy(), decimals=3))
        print('-----')
        # temp = torch.max(temp0*torch.exp(-ANNEAL_RATE*ep), temp_min)  # anneal the temp once every 100 iters? (Jang et al does every 1000 iters)
      temp = torch.max(temp0*torch.exp(-ANNEAL_RATE*ep), temp_min)  # anneal the temp once every 100 iters? (Jang et al does every 1000 iters)
      
      if early_stopped: break
      ##############################################
      
      # temp = torch.max(temp0*torch.exp(-ANNEAL_RATE*ep), temp_min)  # anneal the temp after each iter?
    
    ## add encoder, decoder_.... to saved model
    
    if early_stopped:
      # saved_model = opt_model
      LB = opt_LB
      params_x = opt_params['x']; params_xm = opt_params['xm']; params_y = opt_params['y']; params_ym = opt_params['ym']
      params_r = opt_params['r']; params_z = opt_params['z']
    else:
      ### if model hasn't early stopped, then save the final model
      if (learn_r and not Ignorable): saved_model={'NNs_xm': NNs_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r,'encoder':encoder, 'decoders': decoders}
      else: saved_model={'NNs_xm': NNs_xm, 'NN_ym': NN_ym, 'NN_y': NN_y,'encoder':encoder, 'decoders': decoders}
    
    sys.stdout.flush()
    
    # mse_train={'miss_x':mse_train_miss_x,'obs_x':mse_train_obs_x, 'miss_y':mse_train_miss_y,'obs_y':mse_train_obs_y}
    train_params = {'h1':h1, 'h2':h2, 'h3':h3, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'niws_z':niws_z, 'M':M, 'dim_z':dim_z, 'covars_r_x':covars_r_x, 'covars_r_y':covars_r_y, 'n_hidden_layers':n_hidden_layers, 'n_hidden_layers_y': n_hidden_layers_y, 'n_hidden_layers_r':n_hidden_layers_r, 'pre_impute_value':pre_impute_value, 'early_stop_epochs': early_stop_epochs,'temp': temp.item(), 'L1_weight': L1_weight}
    # all_params = {'x': {'mean':mu_x.cpu().data.numpy(), 'scale':scale_x.cpu().data.numpy()}}
    all_params = {}
    if family=="Gaussian": all_params['y'] = {'mean': params_y['mean'], 'scale': params_y['scale']}
    elif family=="Multinomial": all_params['y'] =  {'probs': params_y['probs']}
    elif family=="Poisson": all_params['y'] = {'lambda': params_y['lambda']}
    if not Ignorable: all_params['r'] = {'probs': params_r['probs']}
    if miss_x:
      # all_params['xm'] = loss_fit['params_xm'].cpu().data.numpy()
      all_params['xm']={}
      # all_params['xm']={'real': {'mean': loss_fit['params_xm']['real']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_xm']['real']['scale'].cpu().data.numpy()},
      #                   'count': {'mean': loss_fit['params_xm']['count']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_xm']['count']['scale'].cpu().data.numpy()},
      #                   'cat': {'mean': loss_fit['params_xm']['real']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_xm']['real']['scale'].cpu().data.numpy()}}
      if exists_types[0]:
        all_params['xm']['real'] = {'mean': params_xm['real']['mean'], 'scale': params_xm['real']['scale']}
      if exists_types[1]:
        all_params['xm']['count'] = {'mean': params_xm['count']['mean'], 'scale': params_xm['count']['scale']}
      if exists_types[2]:
        all_params['xm']['pos'] = {'mean': params_xm['pos']['mean'], 'scale': params_xm['pos']['scale']}
      if exists_types[3]:
        all_params['xm']['cat'] = {}
        for ii in range(0,p_cat):
          all_params['xm']['cat']['probs'] = params_xm['cat'][ii]
      
    all_params['x']={}
    if exists_types[0]:
      all_params['x']['real'] = {'mean': params_x['real']['mean'], 'scale': params_x['real']['scale']}
    if exists_types[1]:
      all_params['x']['count'] = {'mean': params_x['count']['mean'], 'scale': params_x['count']['scale']}
      # all_params['x']['count'] = {'lambda': loss_fit['params_x']['count']['lambda']}
    if exists_types[2]:
      all_params['x']['pos'] = {'mean': params_x['pos']['mean'], 'scale': params_x['pos']['scale']}
    if exists_types[3]:
      all_params['x']['cat'] = {}
      for ii in range(0,p_cat):
        all_params['x']['cat']['probs'] = params_x['cat'][ii]
    
    if miss_y: all_params['ym'] = {'mean': params_ym['mean'],'scale': params_ym['scale']}
    
    return {'train_params':train_params, 'all_params':all_params, 'loss_fit':loss_fit,'impute_fit':impute_fit,'saved_model': saved_model,'LB': LB,'LB_epoch': LB_epoch,'time_train': time_train,'time_impute': time_impute, 'xhat': xhat, 'yhat':yhat, 'yfull':yfull, 'mask_x': mask_x, 'mask_y':mask_y, 'norm_means_x':norm_means_x, 'norm_sds_x':norm_sds_x,'norm_mean_y':norm_mean_y, 'norm_sd_y':norm_sd_y,'val_LB': val_LB,'early_stop_epochs': early_stop_epochs, 'early_stopped':early_stopped}
  else:
    hf = h5py.File(dir_name + '/samples.h5', 'w')
    # temp=temp_min
    temp = torch.tensor(test_temp, device="cuda:0", dtype=torch.float64)
    # validating (hyperparameter values) or testing
    # mu_x = saved_model['mu_x']; scale_x = saved_model['scale_x']
    if (miss_x): NNs_xm=saved_model['NNs_xm']
    if (miss_y): NN_ym=saved_model['NN_ym']
    NN_y=saved_model['NN_y']
    if (learn_r and not Ignorable): NN_r=saved_model['NN_r']
    
    encoder=saved_model['encoder']
    decoders = saved_model['decoders']

    if not draw_miss: cuda_xfull = torch.from_numpy(xfull).float().cuda(); cuda_yfull = torch.from_numpy(yfull).float().cuda()
    else: cuda_xfull = None; cuda_yfull = None

    for ep in range(1,n_epochs):
      if (miss_x):
        if exists_types[0]: NNs_xm['real'].zero_grad()
        if exists_types[1]: NNs_xm['count'].zero_grad()
        if exists_types[2]: NNs_xm['pos'].zero_grad()
        if exists_types[3]:
          for ii in range(0,p_cat):
            NNs_xm['cat'][ii].zero_grad()
      if (miss_y): NN_ym.zero_grad()
      NN_y.zero_grad()
      if (learn_r and not Ignorable): NN_r.zero_grad()
      
      encoder.zero_grad()
      if exists_types[0]: decoders['real'].zero_grad()
      if exists_types[1]: decoders['count'].zero_grad()
      if exists_types[2]: decoders['pos'].zero_grad()
      if exists_types[3]: 
        for ii in range(0,p_cat):
          decoders['cat'][ii].zero_grad()

      # Validation set is much smaller, so including all observations should be fine?
      if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      else: torch_covars_miss = None
      
      #### TAKEN FROM training --> batch the loss computation
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_xfull = np.array_split(xfull[perm,],n/bs)
      batches_x = np.array_split(xhat_0[perm,], n/bs)
      batches_yfull = np.array_split(yfull[perm,],n/bs)
      batches_y = np.array_split(yhat_0[perm,], n/bs)
      batches_mask_x = np.array_split(mask_x[perm,], n/bs)     # only mask for x. for y --> include as a new mask
      batches_mask_y = np.array_split(mask_y[perm,], n/bs)     # only mask for x. for y --> include as a new mask
      batches_loss = []
      if covars: batches_covar = np.array_split(covars_miss[perm,], n/bs)
      splits = np.array_split(perm,n/bs)
      t0_train=time.time()
      for it in range(len(batches_x)):
        if (not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
        else: b_xfull = None; b_yfull = None
        b_x = torch.from_numpy(batches_x[it]).float().cuda()
        b_y = torch.from_numpy(batches_y[it]).float().cuda()
        b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
        b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None
        
        optimizer.zero_grad()
        if miss_x: 
          if exists_types[0]: NNs_xm['real'].zero_grad()
          if exists_types[1]: NNs_xm['count'].zero_grad()
          if exists_types[2]: NNs_xm['pos'].zero_grad()
          if exists_types[3]:
            for ii in range(0,p_cat):
              NNs_xm['cat'][ii].zero_grad()
        
        if miss_y: NN_ym.zero_grad()
        NN_y.zero_grad()
        
        if (learn_r and not Ignorable): NN_r.zero_grad()
        encoder.zero_grad()
        if exists_types[0]: decoders['real'].zero_grad()
        if exists_types[1]: decoders['count'].zero_grad()
        if exists_types[2]: decoders['pos'].zero_grad()
        if exists_types[3]:
          for ii in range(0,p_cat):
            decoders['cat'][ii].zero_grad()
        #mu_x.zero_grad(); scale_x.zero_grad()

        loss_fit = compute_loss(iota_xfull=b_xfull, iota_yfull=b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, temp=temp)
        # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,temp
        temp_params_x, temp_params_xm, temp_params_ym, temp_params_y, temp_params_r, temp_params_z = loss_fit['params_x'], loss_fit['params_xm'], loss_fit['params_ym'], loss_fit['params_y'], loss_fit['params_r'], loss_fit['params_z']

        loss=loss_fit['neg_bound']
        # torch.autograd.set_detect_anomaly(True)
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        
        loss_fit.pop("neg_bound")
        
        if exists_types[0]:
          params_x['real']['mean'][splits[it],:] = temp_params_x['real']['mean']; params_x['real']['scale'][splits[it],:] = temp_params_x['real']['scale']
          if miss_x: params_xm['real']['mean'][splits[it],:] = temp_params_xm['real']['mean']; params_xm['real']['scale'][splits[it],:] = temp_params_xm['real']['scale']
        if exists_types[1]:
          params_x['count']['mean'][splits[it],:] = temp_params_x['count']['mean']; params_x['count']['scale'][splits[it],:] = temp_params_x['count']['scale']
          # params_x['count']['lambda'][splits[it],:] = temp_params_x['count']['lambda']
          if miss_x: params_xm['count']['mean'][splits[it],:] = temp_params_xm['count']['mean']; params_xm['count']['scale'][splits[it],:] = temp_params_xm['count']['scale']
        if exists_types[2]:
          params_x['pos']['mean'][splits[it],:] = temp_params_x['pos']['mean']; params_x['pos']['scale'][splits[it],:] = temp_params_x['pos']['scale']
          if miss_x: params_xm['pos']['mean'][splits[it],:] = temp_params_xm['pos']['mean']; params_xm['pos']['scale'][splits[it],:] = temp_params_xm['pos']['scale']
        if exists_types[3]:
          for ii in range(0, p_cat):
            params_x['cat'][ii][splits[it],:] = temp_params_x['cat'][ii]
            if miss_x: params_xm['cat'][ii][splits[it],:] = temp_params_xm['cat'][ii]
        

        if family=="Gaussian":
          params_y['mean'][splits[it],:] = temp_params_y['mean']; params_y['scale'][splits[it],:] = temp_params_y['scale']
          if miss_y: params_ym['mean'][splits[it],:] = temp_params_ym['mean']; params_ym['scale'][splits[it],:] = temp_params_ym['scale']
        elif family=="Poisson":
          params_y['lambda'][splits[it],:] = temp_params_y['lambda']
          if miss_y: params_ym['mean'][splits[it],:] = temp_params_ym['mean']; params_ym['scale'][splits[it],:] = temp_params_ym['scale']
        elif family=="Multinomial":
          params_y['probs'][splits[it],:] = temp_params_y['probs']
          if miss_y: params_ym['probs'][splits[it],:] = temp_params_ym['probs']

        if not Ignorable: params_r['probs'][splits[it],:] = temp_params_r['probs']
        params_z['mean'][splits[it],:] = temp_params_z['mean']; params_z['scale'][splits[it],:] = temp_params_z['scale']
        # loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})
      total_loss = np.sum(batches_loss)
      
      if arch=="VAE":
        # LB = -total_loss/(niws_z*M**(int(miss_x) + int(miss_y))*n)
        LB = -total_loss/(niws_z*M*n)
      elif arch=="IWAE":
        # LB = -total_loss/(niws_z*M**(int(miss_x) + int(miss_y))*n) + np.log(niws_z) + (int(miss_x) + int(miss_y)) * np.log(M) # miss_x^2 + miss_y^2 = 1 if one miss, 2 if both miss     # redundant: + np.log(M)*(miss_x and miss_y)
        LB = -total_loss/(niws_z*M*n) + np.log(niws_z) + np.log(M) # miss_x^2 + miss_y^2 = 1 if one miss, 2 if both miss     # redundant: + np.log(M)*(miss_x and miss_y)
      
      time_train=np.append(time_train,time.time()-t0_train)
      
      
      if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      else: torch_covars_miss = None
      
      
      
      
      ########## MINIBATCHED THIS (above) #########
      # loss_fit=compute_loss(iota_xfull = cuda_xfull, iota_yfull = cuda_yfull, iota_x = torch.from_numpy(xhat_0).float().cuda(), iota_y = torch.from_numpy(yhat_0).float().cuda(), mask_x = torch.from_numpy(mask_x).float().cuda(), mask_y = torch.from_numpy(mask_y).float().cuda(), covar_miss = torch_covars_miss, temp=temp_min)
      #LB=(-np.log(K) - np.log(M) - loss_fit['neg_bound'].cpu().data.numpy())  
      # LB=(-loss_fit['neg_bound'].cpu().data.numpy())   
      torch.cuda.empty_cache()  # free up gpu memory
      
      t0_impute=time.time()
      batches_xfull = np.array_split(xfull,n/impute_bs)
      batches_x = np.array_split(xhat_0, n/impute_bs)
      batches_y = np.array_split(yfull, n/impute_bs)
      batches_mask_x = np.array_split(mask_x, n/impute_bs)
      batches_mask_y = np.array_split(mask_y, n/impute_bs)
      if covars: batches_covar = np.array_split(covars_miss, n/impute_bs)
      splits = np.array_split(range(n),n/impute_bs)
      if save_imps and (miss_x or miss_y):
        if miss_x and miss_y:
          # all_imp_weights = np.empty([M*M,n])
          # all_xms = np.empty([M*M,np.sum(Rx==0)])
          # all_yms = np.empty([M*M,np.sum(Ry==0)])
          all_imp_weights = np.empty([niws_z*niws_z,n])
          all_xms = np.empty([niws_z*niws_z,np.sum(Rx==0)])
          all_yms = np.empty([niws_z*niws_z,np.sum(Ry==0)])
        else:
          # all_imp_weights = np.empty([M,n])
          # all_xms = np.empty([M,np.sum(Rx==0)])
          # all_yms = np.empty([M,np.sum(Ry==0)])
          all_imp_weights = np.empty([niws_z,n])
          all_xms = np.empty([niws_z,np.sum(Rx==0)])
          all_yms = np.empty([niws_z,np.sum(Ry==0)])
        
        if miss_x: idsx1, idsx2 = np.where(Rx==0); hf.create_dataset("miss_X", data=np.stack((idsx1,idsx2),axis=1), compression="gzip", compression_opts=9)
        if miss_y: idsy = np.where(Ry==0); hf.create_dataset("miss_Y", data=idsy, compression="gzip", compression_opts=9)
      
      for it in range(len(batches_x)):
        if (not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
        else: b_xfull = None; b_yfull=None
        b_x = torch.from_numpy(batches_x[it]).float().cuda()
        b_y = torch.from_numpy(batches_y[it]).float().cuda()
        b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
        b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None
        impute_fit=impute(iota_xfull = b_xfull, iota_yfull = b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, niws_z=niws_z, temp=temp)
        # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,L,temp

        if miss_x:
          b_xhat = xhat[splits[it],:]
          b_xhat[~batches_mask_x[it]] = impute_fit['xm'].cpu().data.numpy()[~batches_mask_x[it]]       # just missing impute
          xhat[splits[it],:] = b_xhat
          xms = impute_fit['xms'].cpu().data.numpy() # [M,bs,p] for miss_x, miss_y, or neither. [M*M,bs,p] for both
        if miss_y:
          b_yhat = yhat[splits[it],:]
          b_yhat[~batches_mask_y[it]] = impute_fit['ym'].cpu().data.numpy()[~batches_mask_y[it]]       # just missing impute
          yhat[splits[it],:] = b_yhat
          yms = impute_fit['yms'].cpu().data.numpy() # [M,bs,1] for miss_x, miss_y, or neither. [M*M,bs,1] for both
        imp_weights = impute_fit['imp_weights'].cpu().data.numpy()
        if save_imps and (miss_x or miss_y):
          for ii in range(0, xms.shape[0]):
            # print(ii)
            # # print(np.isin(idsx1, splits[it]))
            # print(xms.shape)
            # print(b_mask_x.shape)
            # print(b_mask_x[:4,:4])
            # print(all_xms.shape)
            # print(all_yms.shape)
            sys.stdout
            if miss_x: all_xms[ii,np.isin(idsx1, splits[it])] = xms[ii,:,:][b_mask_x.cpu().data.numpy()==0]
            if miss_y: all_yms[ii,np.isin(idsy, splits[it])] = yms[ii,:,0][b_mask_y.cpu().data.numpy()==0]
          all_imp_weights[:,splits[it]] = imp_weights
      
      torch.cuda.empty_cache()  # free up gpu memory
      if save_imps and (miss_x or miss_y):
        for ii in range(0, xms.shape[0]):  # xms.shape[0] = yms.shape[0] = M*M or M
          print("Saving sample #" + str(ii))
          if miss_x: hf.create_dataset("Xm"+str(ii), data=all_xms[ii,:], compression="gzip", compression_opts=9)
          if miss_y: hf.create_dataset("Ym"+str(ii), data=all_yms[ii,:], compression="gzip", compression_opts=9)
        # np.savetxt(dir_name + "/IWs.csv",all_imp_weights, delimiter=",")  # should be L x n or M*L x n
        print("Saving IWs")
        hf.create_dataset("IWs", data=all_imp_weights, compression="gzip", compression_opts=9)
      hf.close()
      print("Saving imputed datasets complete")

      time_impute=np.append(time_impute,time.time()-t0_impute)
      
      ########## MINIBATCHED ABOVE #########
      # impute_fit=impute(iota_xfull = cuda_xfull, iota_yfull = cuda_yfull, iota_x = torch.from_numpy(xhat_0).float().cuda(), iota_y = torch.from_numpy(yhat_0).float().cuda(), mask_x = torch.from_numpy(mask_x).float().cuda(), mask_y = torch.from_numpy(mask_y).float().cuda(), covar_miss = torch_covars_miss,L=L,temp=temp_min)
      # # impute xm:
      # if miss_x: xhat[~mask_x] = impute_fit['xm'].cpu().data.numpy()[~mask_x]
      # # impute ym:
      # if miss_y: yhat[~mask_y] = impute_fit['ym'].cpu().data.numpy()[~mask_y]

      # err_x = mse(xhat, xfull, mask_x)
      # err_y = mse(yhat, yfull, mask_y)
      # 
      # print('Observed MSE x:  %g' %err_x['obs'])   # these aren't reconstructed/imputed
      # print('Missing MSE x:  %g' %err_x['miss'])
      # print('Observed MSE y:  %g' %err_y['obs'])   # these aren't reconstructed/imputed
      # print('Missing MSE y:  %g' %err_y['miss'])
      # print('-----')
      # if exists_types[0]: err_x_real = mse(((xhat*norm_sds_x) + norm_means_x)[ids_real], ((xfull*norm_sds_x)+norm_means_x)[ids_real], mask_x[:,ids_real])
      # if exists_types[1]: err_x_count = mse(((xhat*norm_sds_x) + norm_means_x)[ids_count], ((xfull*norm_sds_x)+norm_means_x)[ids_count], mask_x[:,ids_count])
      errs = {}
      if exists_types[0]: err_x_real = mse(xhat[:,ids_real], xfull[:,ids_real], mask_x[:,ids_real]); errs['real']=err_x_real
      else: errs['real']={'miss': np.nan, 'obs': np.nan}
      if exists_types[1]: err_x_count = mse(xhat[:,ids_count], xfull[:,ids_count], mask_x[:,ids_count]); errs['count']=err_x_count
      else: errs['count']={'miss': np.nan, 'obs': np.nan}
      if exists_types[2]: err_x_pos = mse(xhat[:,ids_pos], xfull[:,ids_pos], mask_x[:,ids_pos]); errs['pos']=err_x_pos
      else: errs['pos']={'miss': np.nan, 'obs': np.nan}
      if exists_types[3]: 
        err_x_cat0 = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat]) #err_x_cat = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat])
        err_x_cat1 = pred_acc(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat],Cs) #err_x_cat = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat])
        if np.sum(np.isnan(xfull[:,ids_cat])) == 0:
          err_x_cat1 = pred_acc(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat],Cs) #err_x_cat = mse(xhat[:,ids_cat],xfull[:,ids_cat],mask_x[:,ids_cat])
        else:
          err_x_cat1 = {'miss': np.nan, 'obs': np.nan}
        errs['cat0'] = err_x_cat0
        errs['cat1'] = err_x_cat1
      else:
        errs['cat0'] = {'miss': np.nan, 'obs': np.nan}
        errs['cat1'] = {'miss': np.nan, 'obs': np.nan}
      if family=="Multinomial": err_y = pred_acc(yhat*norm_sd_y+norm_mean_y, yfull*norm_sd_y+norm_mean_y, mask_y, C)
      else: err_y = mse(yhat*norm_sd_y+norm_mean_y, yfull*norm_sd_y+norm_mean_y, mask_y)
      errs['y'] = err_y
      
      if exists_types[0]:
        print('Observed MSE x_real:  %g' %err_x_real['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE x_real:  %g' %err_x_real['miss'])
      if exists_types[1]:
        print('Observed MSE x_count (log-transformed):  %g' %err_x_count['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE x_count (log-transformed):  %g' %err_x_count['miss'])
      if exists_types[2]:
        print('Observed MSE x_pos:  %g' %err_x_pos['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE x_pos:  %g' %err_x_pos['miss'])
      if exists_types[3]:
        print('Observed MSE x_cat:  %g' %err_x_cat0['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE x_cat:  %g' %err_x_cat0['miss'])
        print('Observed Pred_Acc x_cat:  %g' %err_x_cat1['obs'])   # these aren't reconstructed/imputed
        print('Missing Pred_Acc x_cat:  %g' %err_x_cat1['miss'])
      print('Observed MSE/Pred_Acc y (Gaussian/Multinomial):  %g' %err_y['obs'])   # these aren't reconstructed/imputed
      print('Missing MSE/Pred_Acc y (Gaussian/Multinomial):  %g' %err_y['miss'])
      print('-----')
    
        # revert cat vars back to the way they were in xhat, xfull, and mask_x
    ### REVERT data_types_x = c(r, r, r, count, count, count, cat*Cs[0], cat*Cs[1])
    ### USE data_types_x_0: original data_types_x = c(r, r, r, cat, cat, count, count, count)

    #xhat
    #Cs
      
    # raise Exception("TEST")
    xhat0 = np.empty([xhat.shape[0], len(data_types_x_0)])
    mask_x0 = np.empty([mask_x.shape[0], len(data_types_x_0)])
    covars_r_x0 = np.empty([len(data_types_x_0)])
    i_real=0; i_count=0; i_cat=0; i_pos=0; C0=0
    
    # undo normalization (scaling):
    xhat = (xhat*norm_sds_x) + norm_means_x
    if family != "Multinomial":
      yhat = (yhat*norm_sd_y) + norm_mean_y
    
    for i in range(0,len(data_types_x_0)):
      if data_types_x_0[i]=="real":
        xhat0[:,i] = xhat[:,np.where(ids_real)[0][i_real]]
        mask_x0[:,i] = mask_x[:,np.where(ids_real)[0][i_real]]
        covars_r_x0[i] = covars_r_x[np.where(ids_real)[0][i_real]]
        i_real = i_real+1
      elif data_types_x_0[i]=="count":
        # xhat0[:,i] = np.exp(xhat[:,np.where(ids_count)[0][i_count]])
        xhat0[:,i] = xhat[:,np.where(ids_count)[0][i_count]]
        mask_x0[:,i] = mask_x[:,np.where(ids_count)[0][i_count]]
        covars_r_x0[i] = covars_r_x[np.where(ids_count)[0][i_count]]
        i_count=i_count+1
      elif data_types_x_0[i]=="pos":
        xhat0[:,i] = xhat[:,np.where(ids_pos)[0][i_pos]]
        mask_x0[:,i] = mask_x[:,np.where(ids_pos)[0][i_pos]]
        covars_r_x0[i] = covars_r_x[np.where(ids_pos)[0][i_pos]]
        i_pos = i_pos+1
      elif data_types_x_0[i]=="cat":
        idd = np.where(ids_cat)[0][int(C0*i_cat):int(C0*i_cat+Cs[i_cat])]   # should give indices for dummy vars pertaining to respective cat. var
        xhat0[:,i] = np.argmax(xhat[:,idd], axis=1) + 1      # above line
        mask_x0[:,i] = mask_x[:,idd[0]]   # can be max or min or anything --> should still be the same value: 0 or 1
        covars_r_x0[i] = int(covars_r_x[idd[0]])
        print("data_types_x_0");print(data_types_x_0)
        print("len(data_types_x_0)");print(len(data_types_x_0))
        print("covars_r_x"); print(covars_r_x)
        print("idd"); print(idd)
        print("covars_r_x0"); print(covars_r_x0)
        C0=Cs[i_cat]
        i_cat = i_cat+1
    xhat=xhat0
    mask_x=mask_x0
    data_types_x=data_types_x_0
    covars_r_x=covars_r_x0
    
    ### what if y was count? yhat?
    
    # mse_test={'miss_x':err_x['miss'],'obs_x':err_x['obs'], 'miss_y':err_y['miss'],'obs_y':err_y['obs']}
    # if (learn_r): saved_model={'NN_xm': NN_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r, 'mu_x':mu_x, 'scale_x':scale_x}
    # else: saved_model={'NN_xm': NN_xm, 'NN_ym':NN_ym, 'NN_y': NN_y, 'mu_x':mu_x, 'scale_x':scale_x}
    # all_params = {'x': {'mean':mu_x.cpu().data.numpy(), 'scale':scale_x.cpu().data.numpy()}}
    # if (learn_r): saved_model={'NN_xm': NN_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r}
    # else: saved_model={'NN_xm': NN_xm, 'NN_ym':NN_ym, 'NN_y': NN_y}
    if (learn_r and not Ignorable): saved_model={'NNs_xm': NNs_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r,'encoder':encoder}
    else: saved_model={'NNs_xm': NNs_xm, 'NN_ym': NN_ym, 'NN_y': NN_y,'encoder':encoder}
    
    saved_model['decoders'] = decoders
    sys.stdout.flush()
    
    all_params = {}
    if family=="Gaussian": all_params['y'] = {'mean': params_y['mean'], 'scale': params_y['scale']}
    elif family=="Multinomial": all_params['y'] =  {'probs': params_y['probs']}
    elif family=="Poisson": all_params['y'] = {'lambda': params_y['lambda']}
    if not Ignorable: all_params['r'] = {'probs': params_r['probs']}
    if miss_x:
      # all_params['xm'] = loss_fit['params_xm'].cpu().data.numpy()
      all_params['xm']={}
      # all_params['xm']={'real': {'mean': loss_fit['params_xm']['real']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_xm']['real']['scale'].cpu().data.numpy()},
      #                   'count': {'mean': loss_fit['params_xm']['count']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_xm']['count']['scale'].cpu().data.numpy()},
      #                   'cat': {'mean': loss_fit['params_xm']['real']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_xm']['real']['scale'].cpu().data.numpy()}}
      if exists_types[0]:
        all_params['xm']['real'] = {'mean': params_xm['real']['mean'], 'scale': params_xm['real']['scale']}
      if exists_types[1]:
        all_params['xm']['count'] = {'mean': params_xm['count']['mean'], 'scale': params_xm['count']['scale']}
      if exists_types[2]:
        all_params['xm']['pos'] = {'mean': params_xm['pos']['mean'], 'scale': params_xm['pos']['scale']}
      if exists_types[3]:
        all_params['xm']['cat'] = {}
        for ii in range(0,p_cat):
          all_params['xm']['cat']['probs'] = params_xm['cat'][ii]

    all_params['x']={}
    if exists_types[0]:
      all_params['x']['real'] = {'mean': params_x['real']['mean'], 'scale': params_x['real']['scale']}
    if exists_types[1]:
      all_params['x']['count'] = {'mean': params_x['count']['mean'], 'scale': params_x['count']['scale']}
      # all_params['x']['count'] = {'lambda': loss_fit['params_x']['count']['lambda']}
    if exists_types[2]:
      all_params['x']['pos'] = {'mean': params_x['pos']['mean'], 'scale': params_x['pos']['scale']}
    if exists_types[3]:
      all_params['x']['cat'] = {}
      for ii in range(0,p_cat):
        all_params['x']['cat']['probs'] = params_x['cat'][ii]

    if miss_y: all_params['ym'] = {'mean': params_ym['mean'],'scale': params_ym['scale']}
    train_params = {'ids_real':ids_real, 'ids_count':ids_count, 'ids_cat':ids_cat, 'exists_types': exists_types,'h1':h1, 'h2':h2, 'h3':h3, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'niws_z':niws_z, 'M':M, 'dim_z':dim_z, 'covars_r_x':covars_r_x, 'covars_r_y':covars_r_y, 'n_hidden_layers':n_hidden_layers, 'n_hidden_layers_y':n_hidden_layers_y, 'n_hidden_layers_r':n_hidden_layers_r, 'pre_impute_value':pre_impute_value, 'temp': temp.item(), 'L1_weight': L1_weight}
    
    if isinstance(NN_y[0].bias,type(None)): w0 = 0
    else: w0 = (NN_y[0].bias).cpu().data.numpy()
    w = (NN_y[0].weight).cpu().data.numpy()
    return {'w0':w0,'w':w,'train_params':train_params,'loss_fit':loss_fit,'impute_fit':impute_fit,'saved_model': saved_model,'all_params':all_params,'LB': LB,'time_impute': time_impute,'xhat': xhat, 'yhat':yhat, 'xfull': xfull, 'yfull':yfull, 'mask_x': mask_x, 'mask_y':mask_y, 'norm_means_x':norm_means_x, 'norm_sds_x':norm_sds_x,'norm_mean_y':norm_mean_y, 'norm_sd_y':norm_sd_y,'errs':errs}
