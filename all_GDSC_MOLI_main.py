import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk
import torch.nn.functional as F
import random

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime

from utils import AllTripletSelector,AllPositivePairSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from models import Supervised_Encoder, Classifier, OnlineTriplet, OnlineTestTriplet
from Super_FELT_utils import read_files_for_only_GDSC, processing_files_for_only_GDSC_without_FS
from matplotlib import pyplot as plt

torch.manual_seed(42)
drugs = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])


def main(start,end,gpu_num,drugs,save_results_to,work_num,hyperparameters_set):

    marg = hyperparameters_set['mrg']
    lrE = hyperparameters_set['lre']
    lrM = hyperparameters_set['lrm']
    lrC = hyperparameters_set['lrc']
    lrCL = hyperparameters_set['lrCL']

    mb_size = hyperparameters_set['mbs']
    OE_dim = hyperparameters_set['hdm']
    OM_dim = hyperparameters_set['hdm']
    OC_dim = hyperparameters_set['hdm']
    ICP_dim = OE_dim+OM_dim+OC_dim
    OCP_dim = ICP_dim


    epoch = hyperparameters_set['epch']
    lam = hyperparameters_set['lam']

    E_dr = 0.5
    C_dr = 0.5
    Cwd = hyperparameters_set['wd']
    Ewd = 0

    torch.cuda.set_device(gpu_num)
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    #triplet_selector = RandomNegativeTripletSelector(marg)
    triplet_selector2 = AllTripletSelector()
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)
    BCE_loss_fun = torch.nn.BCELoss()

    skf = StratifiedKFold(n_splits=5, random_state=42)

    all_result = []

    for i in range(start,end):
        drug = drugs[i]
        origin_GDSCE,origin_GDSCM,origin_GDSCC,origin_GDSCR = read_files_for_only_GDSC(data_dir,drug)


        if len(origin_GDSCE) != 0:
            GDSCE,GDSCM,GDSCC,GDSCR = processing_files_for_only_GDSC_without_FS(origin_GDSCE,origin_GDSCM,origin_GDSCC,origin_GDSCR)
            GDSCE = GDSCE.apply(pd.to_numeric)
            GDSCM = GDSCM.apply(pd.to_numeric)
            GDSCC = GDSCC.apply(pd.to_numeric)
            print(drug)
            record_list = []
            total_train_auc = []
            total_val_auc = []
            total_test_auc = []

            max_iter = 5
            for iters in range(max_iter):
                k = 0
                GDSCE,GDSCM,GDSCC,GDSCR = shuffle(GDSCE,GDSCM,GDSCC,GDSCR)
                Y = GDSCR['response'].values
                Y = sk.LabelEncoder().fit_transform(Y)

                torch.cuda.empty_cache()
                for train_index, test_index in skf.split(GDSCE.values, Y):
                    k = k + 1
                    """
                        x data is only GDSC
                    """
                    X_trainE = GDSCE.values[train_index,:]
                    X_testE =  GDSCE.values[test_index,:]
                    X_trainM = GDSCM.values[train_index,:]
                    X_testM = GDSCM.values[test_index,:]
                    X_trainC = GDSCC.values[train_index,:]
                    X_testC = GDSCC.values[test_index,:]
                    Y_train = Y[train_index]
                    Y_test = Y[test_index]


                    scalerGDSC = sk.StandardScaler()
                    scalerGDSC.fit(X_trainE)
                    X_trainE = scalerGDSC.transform(X_trainE)
                    X_testE = scalerGDSC.transform(X_testE)

                    X_trainE, X_valE, X_trainM, X_valM, X_trainC, X_valC, Y_train, Y_val \
                        = train_test_split(X_trainE,X_trainM,X_trainC, Y_train, test_size=0.2, random_state=42,stratify=Y_train)

                    TX_testE = torch.FloatTensor(X_testE).to(device)
                    TX_testM = torch.FloatTensor(X_testM).to(device)
                    TX_testC = torch.FloatTensor(X_testC).to(device)
                    TY_test = torch.FloatTensor(Y_test.astype(int)).to(device)

                    TX_valE = torch.FloatTensor(X_valE).to(device)
                    TX_valM = torch.FloatTensor(X_valM).to(device)
                    TX_valC = torch.FloatTensor(X_valC).to(device)
                    TY_val = torch.FloatTensor(Y_val.astype(int)).to(device)

                    #Train
                    class_sample_count = np.array([len(np.where(Y_train==t)[0]) for t in np.unique(Y_train)])
                    weight = 1. / class_sample_count
                    samples_weight = np.array([weight[t] for t in Y_train])

                    samples_weight = torch.from_numpy(samples_weight)
                    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

                    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                                     torch.FloatTensor(X_trainC), torch.FloatTensor(Y_train.astype(int)))

                    trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size= mb_size, shuffle=False, num_workers= 0 , sampler = sampler)

                    n_sampE, IE_dim = X_trainE.shape
                    n_sampM, IM_dim = X_trainM.shape
                    n_sampC, IC_dim = X_trainC.shape

                    cost_tr = []
                    auc_tr = []
                    cost_val = []
                    auc_val = []

                    torch.cuda.manual_seed_all(42)

                    E_Supervised_Encoder = Supervised_Encoder(IE_dim,OE_dim,E_dr)
                    M_Supervised_Encoder = Supervised_Encoder(IM_dim,OM_dim,E_dr)
                    C_Supervised_Encoder = Supervised_Encoder(IC_dim,OC_dim,E_dr)
                    E_Supervised_Encoder.to(device)
                    M_Supervised_Encoder.to(device)
                    C_Supervised_Encoder.to(device)

                    E_optimizer = optim.Adagrad(E_Supervised_Encoder.parameters(), lr=lrE,weight_decay = Ewd)
                    M_optimizer = optim.Adagrad(M_Supervised_Encoder.parameters(), lr=lrM, weight_decay = Ewd)
                    C_optimizer = optim.Adagrad(C_Supervised_Encoder.parameters(), lr=lrC, weight_decay = Ewd)

                    TripSel = OnlineTestTriplet(marg, triplet_selector2)

                    Clas = Classifier(OCP_dim,1,C_dr)
                    Clas.to(device)
                    Cl_optimizer = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = Cwd)

                    for ep in range(epoch):
                        epoch_cost = 0
                        epoch_auc_list = []
                        num_minibatches = int(n_sampE / mb_size)
                        flag = 0
                        E_Supervised_Encoder.train()
                        M_Supervised_Encoder.train()
                        C_Supervised_Encoder.train()
                        Clas.train()
                        for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):

                            if torch.mean(target)!=0. and torch.mean(target)!=1.and len(target)>2:
                                dataE = dataE.to(device)
                                dataM = dataM.to(device)
                                dataC = dataC.to(device)
                                target = target.to(device)
                                encoded_E = E_Supervised_Encoder(dataE)
                                encoded_M = M_Supervised_Encoder(dataM)
                                encoded_C = C_Supervised_Encoder(dataC)

                                intergrated_omics = torch.cat((encoded_E, encoded_M, encoded_C), 1)
                                intergrated_omics = F.normalize(intergrated_omics, p=2, dim=0)
                                Pred = Clas(intergrated_omics)

                                y_true = target.view(-1,1).cpu()

                                Triplets_list = TripSel(intergrated_omics, target)

                                triplet_loss = trip_loss_fun(intergrated_omics[Triplets_list[:,0],:],intergrated_omics[Triplets_list[:,1],:],intergrated_omics[Triplets_list[:,2],:])

                                loss = lam * triplet_loss + BCE_loss_fun(Pred,target.view(-1,1))

                                y_pred = Pred.cpu()
                                AUC = roc_auc_score(y_true.detach().numpy(),y_pred.detach().numpy())

                                E_optimizer.zero_grad()
                                M_optimizer.zero_grad()
                                C_optimizer.zero_grad()
                                Cl_optimizer.zero_grad()

                                loss.backward()

                                E_optimizer.step()
                                M_optimizer.step()
                                C_optimizer.step()
                                Cl_optimizer.step()

                                epoch_cost = epoch_cost + (loss / num_minibatches)
                                epoch_auc_list.append(AUC)
                                flag =1

                        if flag == 1:
                            cost_tr.append(torch.mean(epoch_cost))
                            auc_tr.append(np.mean(epoch_auc_list))
                            total_train_auc.append(np.mean(epoch_auc_list))
                            print('Iter-{}; Total loss: {:.4}'.format(ep, loss))

                        with torch.no_grad():
                            E_Supervised_Encoder.eval()
                            M_Supervised_Encoder.eval()
                            C_Supervised_Encoder.eval()
                            Clas.eval()

                            encoded_val_E = E_Supervised_Encoder(TX_valE)
                            encoded_val_M = M_Supervised_Encoder(TX_valM)
                            encoded_val_C = C_Supervised_Encoder(TX_valC)

                            intergrated_val_omics = torch.cat((encoded_val_E, encoded_val_M, encoded_val_C), 1)
                            val_Pred = Clas(intergrated_val_omics)

                            triplet_loss = trip_loss_fun(intergrated_val_omics[Triplets_list[:,0],:],
                                                         intergrated_val_omics[Triplets_list[:,1],:],
                                                         intergrated_val_omics[Triplets_list[:,2],:])

                            val_loss =  lam * triplet_loss + BCE_loss_fun(val_Pred,TY_val.view(-1,1))

                            val_y_true = TY_val.view(-1,1).cpu()
                            val_y_pred = val_Pred.cpu()

                            val_AUC = roc_auc_score(val_y_true.detach().numpy(),val_y_pred.detach().numpy())
                            print("val_AUC: ",val_AUC)


                    E_Supervised_Encoder.eval()
                    M_Supervised_Encoder.eval()
                    C_Supervised_Encoder.eval()
                    Clas.eval()

                    cost_val.append(val_loss)
                    auc_val.append(val_AUC)
                    total_val_auc.append(val_AUC)

                    """
                        test
                    """
                    encoded_test_E = E_Supervised_Encoder(TX_testE)
                    encoded_test_M = M_Supervised_Encoder(TX_testM)
                    encoded_test_C = C_Supervised_Encoder(TX_testC)
                    intergrated_test_omics = torch.cat((encoded_test_E, encoded_test_M, encoded_test_C), 1)
                    test_Pred = Clas(intergrated_test_omics)

                    test_y_true = TY_test.view(-1,1).cpu()
                    test_y_pred = test_Pred.cpu()

                    test_AUC = roc_auc_score(test_y_true.detach().numpy(),test_y_pred.detach().numpy())
                    total_test_auc.append(test_AUC)


                    test_AUC = roc_auc_score(test_y_true.detach().numpy(),test_y_pred.detach().numpy())
                    total_test_auc.append(test_AUC)

                    title = str(datetime.now())+'iters {}, epoch = {},lam = {}, mb_size = {},  out_dim[1,2,3] = ({},{},{}), marg = {}, lr[E,M,C] = ({}, {}, {}), Cwd = {},Ewd = {}, lrCL = {}, dropout[Supervised_Encoder,classifier]=({},{})'.\
                                  format(iters,epoch,lam, mb_size, OE_dim, OM_dim, OC_dim, marg, lrE, lrM, lrC, Cwd,Ewd, lrCL,E_dr,C_dr)

                    print(title)
                    print("####################sum(auc_tr)/len(auc_tr): ", sum(auc_tr)/len(auc_tr))
                    print("####################sum(auc_val)/len(auc_val): ", sum(auc_val)/len(auc_val))
                    print("####################test_AUC: ", test_AUC)
                    record_list.append([iters,sum(auc_tr)/len(auc_tr),sum(auc_val)/len(auc_val),test_AUC])

                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_train_auc: ", sum(total_train_auc)/len(total_train_auc))
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_val_auc: ", sum(total_val_auc)/len(total_val_auc))
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
                    print()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_train_auc: ", sum(total_train_auc)/len(total_train_auc))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_val_auc: ", sum(total_val_auc)/len(total_val_auc))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
        record_list.append(['total',sum(total_train_auc)/len(total_train_auc),sum(total_val_auc)/len(total_val_auc),sum(total_test_auc)/len(total_test_auc)])
        record_list.append(['mb_size','OE_dim','OM_dim','OC_dim'])
        record_list.append([mb_size,OE_dim,OM_dim,OC_dim])
        record_list.append([title,'','',''])
        record_df = pd.DataFrame(data = record_list,columns = ['iters('+drug+')','avg(aucTrain)','avg(aucValidation)','avg(aucTest)'])
        record_df.to_csv(save_results_to+str(datetime.now())+'_'+drug+'result.txt',sep='\t',index=None)

        all_result.append([drug,np.average(total_train_auc),np.average(total_val_auc),np.average(total_test_auc)])
    all_result.append([list(hyperparameters_set),'','',''])
    df_all = pd.DataFrame(data=all_result,columns=['drug','avg(aucTrain)','avg(aucValidation)','avg(aucTest)'])
    df_all.to_csv(save_results_to+str(datetime.now())+'_all_result.txt',sep='\t',index=None)

data_dir = '/GDSC/'
save_results = '/MOLI_GDSC_results/'

gpu_num = 0
for i in range(8):
    work_num = gpu_num
    ls_mb_size = [37, 30, 64]
    ls_h_dim = [128, 64, 32]
    ls_marg = [0.5, 1, 1.5]
    ls_lr = [0.5, 0.1, 0.05, 0.01]
    ls_epoch = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
    ls_rate = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ls_wd = [0.01, 0.001, 0.1, 0.0001]
    ls_lam = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    mbs = random.choice(ls_mb_size)
    hdm = random.choice(ls_h_dim)
    mrg = random.choice(ls_marg)
    lre = random.choice(ls_lr)
    lrm = random.choice(ls_lr)
    lrc = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    wd = random.choice(ls_wd)
    lam = random.choice(ls_lam)
    hyperparameters_set = {'mbs':mbs,'hdm':hdm,'mrg':mrg,'lre':lre,
                           'lrm':lrm,'lrc':lrc,'lrCL':lrCL,'epch':epch,
                           'wd':wd,'lam':lam}

    start = 0
    end = len(drugs)
    main(start, end, gpu_num,drugs,save_results,work_num,hyperparameters_set)
