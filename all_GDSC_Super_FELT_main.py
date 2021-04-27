import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from datetime import datetime

from utils import AllTripletSelector,AllPositivePairSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from models import Supervised_Encoder, Classifier, OnlineTriplet, OnlineTestTriplet
from Super_FELT_utils import read_files_for_only_GDSC, processing_files_for_only_GDSC


torch.manual_seed(42)
drugs = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])

hyperparameters_set_list = []
hyperparameters_set1 = {'E_dr':0.1, 'C_dr':0.1,'Cwd':0.0,'Ewd':0.0}
hyperparameters_set2 = {'E_dr':0.3, 'C_dr':0.3,'Cwd':0.01,'Ewd':0.01}
hyperparameters_set3 = {'E_dr':0.3, 'C_dr':0.3,'Cwd':0.01,'Ewd':0.05}
hyperparameters_set4 = {'E_dr':0.5, 'C_dr':0.5,'Cwd':0.01,'Ewd':0.01}
hyperparameters_set5 = {'E_dr':0.5, 'C_dr':0.7,'Cwd':0.15,'Ewd':0.1}
hyperparameters_set6 = {'E_dr':0.3, 'C_dr':0.5,'Cwd':0.01,'Ewd':0.01}
hyperparameters_set7 = {'E_dr':0.4, 'C_dr':0.4,'Cwd':0.01,'Ewd':0.01}
hyperparameters_set8 = {'E_dr':0.5, 'C_dr':0.5,'Cwd':0.1,'Ewd':0.1}


hyperparameters_set_list.append(hyperparameters_set1)
hyperparameters_set_list.append(hyperparameters_set2)
hyperparameters_set_list.append(hyperparameters_set3)
hyperparameters_set_list.append(hyperparameters_set4)
hyperparameters_set_list.append(hyperparameters_set5)
hyperparameters_set_list.append(hyperparameters_set6)
hyperparameters_set_list.append(hyperparameters_set7)
hyperparameters_set_list.append(hyperparameters_set8)


def main(start,end,gpu_num,drugs,save_results_to,work_num,hyperparameters_set):
    #common hyperparameters
    marg = 1
    lrE = 0.01
    lrM = 0.01
    lrC = 0.01
    lrCL = 0.01

    mb_size = 55
    OE_dim = 256
    OM_dim = 32
    OC_dim = 64
    In_dim = OE_dim+OM_dim+OC_dim
    Out_dim = In_dim

    E_Supervised_Encoder_epoch = 10
    C_Supervised_Encoder_epoch = 10
    M_Supervised_Encoder_epoch = 10
    Classifier_epoch = 10

    #non-common hyperparameters
    E_dr = hyperparameters_set['E_dr']
    C_dr = hyperparameters_set['C_dr']
    Cwd = hyperparameters_set['Cwd']
    Ewd = hyperparameters_set['Ewd']


    triplet_selector2 = AllTripletSelector()
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)
    BCE_loss_fun = torch.nn.BCELoss()

    skf = StratifiedKFold(n_splits=5, random_state=42)
    torch.cuda.set_device(gpu_num)
    device = torch.device('cuda')


    for i in range(start,end):
        drug = drugs[i]
        origin_GDSCE,origin_GDSCM,origin_GDSCC,origin_GDSCR = read_files_for_only_GDSC(data_dir,drug)


        if len(origin_GDSCE) != 0:
            GDSCE,GDSCM,GDSCC,GDSCR = processing_files_for_only_GDSC(origin_GDSCE,origin_GDSCM,origin_GDSCC,origin_GDSCR)
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
                for train_index, test_index in skf.split(GDSCE.values, Y):
                    torch.cuda.empty_cache()
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
                    Clas = Classifier(Out_dim,1,C_dr)
                    Clas.to(device)
                    Cl_optimizer = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = Cwd)

                    ## train each Supervised_Encoder with thriplet loss
                    pre_loss = 100
                    break_num = 0
                    for e_epoch in range(E_Supervised_Encoder_epoch):
                        E_Supervised_Encoder.train()
                        flag = 0
                        for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                            if torch.mean(target)!=0. and torch.mean(target)!=1. and len(target) > 2:
                                dataE = dataE.to(device)
                                encoded_E = E_Supervised_Encoder(dataE)

                                E_Triplets_list = TripSel(encoded_E, target)
                                E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:,0],:],encoded_E[E_Triplets_list[:,1],:],encoded_E[E_Triplets_list[:,2],:])

                                E_optimizer.zero_grad()
                                E_loss.backward()
                                E_optimizer.step()
                                flag = 1

                        if flag == 1:
                            print('Iter-{}; E_loss: {:.4}'.format(e_epoch,E_loss))
                        with torch.no_grad():
                            E_Supervised_Encoder.eval()

                            encoded_val_E = E_Supervised_Encoder(TX_valE)
                            E_Triplets_list = TripSel(encoded_val_E, TY_val)
                            val_E_loss = trip_loss_fun(encoded_val_E[E_Triplets_list[:,0],:],encoded_val_E[E_Triplets_list[:,1],:],encoded_val_E[E_Triplets_list[:,2],:])

                            print("val_E_loss: ", val_E_loss)
                            if pre_loss <= val_E_loss:
                                break_num +=1

                            if break_num >1:
                                break
                            else:
                                pre_loss = val_E_loss

                    E_Supervised_Encoder.eval()

                    pre_loss = 100
                    break_num = 0
                    for m_epoch in range(M_Supervised_Encoder_epoch):
                        M_Supervised_Encoder.train().to(device)
                        flag = 0
                        for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                            if torch.mean(target)!=0. and torch.mean(target)!=1. and len(target) > 2:
                                dataM = dataM.to(device)
                                encoded_M = M_Supervised_Encoder(dataM)
                                M_Triplets_list = TripSel(encoded_M, target)
                                M_loss = trip_loss_fun(encoded_M[M_Triplets_list[:,0],:],encoded_M[M_Triplets_list[:,1],:],encoded_M[M_Triplets_list[:,2],:])

                                M_optimizer.zero_grad()
                                M_loss.backward()
                                M_optimizer.step()
                                flag = 1

                        if flag == 1:
                            print('Iter-{}; M_loss: {:.4}'.format(m_epoch,M_loss))
                        with torch.no_grad():
                            M_Supervised_Encoder.eval()
                            """
                                test
                            """
                            encoded_val_M = M_Supervised_Encoder(TX_valM)
                            M_Triplets_list = TripSel(encoded_val_M, TY_val)
                            val_M_loss = trip_loss_fun(encoded_val_M[M_Triplets_list[:,0],:],encoded_val_M[M_Triplets_list[:,1],:],encoded_val_M[M_Triplets_list[:,2],:])

                            print("val_M_loss: ", val_M_loss)
                            if pre_loss <= val_M_loss:
                                break_num +=1

                            if break_num >1:
                                break
                            else:
                                pre_loss = val_M_loss

                    M_Supervised_Encoder.eval()

                    pre_loss = 100
                    break_num = 0
                    for c_epoch in range(C_Supervised_Encoder_epoch):
                        C_Supervised_Encoder.train()
                        flag = 0
                        for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                            if torch.mean(target)!=0. and torch.mean(target)!=1. and len(target)>2:
                                dataC = dataC.to(device)
                                encoded_C = C_Supervised_Encoder(dataC)

                                C_Triplets_list = TripSel(encoded_C, target)
                                C_loss = trip_loss_fun(encoded_C[C_Triplets_list[:,0],:],encoded_C[C_Triplets_list[:,1],:],encoded_C[C_Triplets_list[:,2],:])

                                C_optimizer.zero_grad()
                                C_loss.backward()
                                C_optimizer.step()

                                flag = 1

                        if flag == 1:
                            print('Iter-{}; C_loss: {:.4}'.format(c_epoch,C_loss))
                        with torch.no_grad():
                            C_Supervised_Encoder.eval()
                            """
                                inner test
                            """
                            encoded_val_C = C_Supervised_Encoder(TX_testC)
                            C_Triplets_list = TripSel(encoded_val_C, TY_val)
                            val_C_loss = trip_loss_fun(encoded_val_C[C_Triplets_list[:,0],:],encoded_val_C[C_Triplets_list[:,1],:],encoded_val_C[C_Triplets_list[:,2],:])
                            print("val_C_loss: ", val_C_loss)
                            if pre_loss <= val_C_loss:
                                break_num +=1

                            if break_num >1:
                                break
                            else:
                                pre_loss = val_C_loss

                    C_Supervised_Encoder.eval()

                    ## train classifier
                    pre_auc = 0
                    for cl_epoch in range(Classifier_epoch):
                        epoch_cost = 0
                        epoch_auc_list = []
                        num_minibatches = int(n_sampE / mb_size)
                        flag = 0
                        Clas.train()
                        for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):

                            if torch.mean(target)!=0. and torch.mean(target)!=1.:
                                dataE = dataE.to(device)
                                dataM = dataM.to(device)
                                dataC = dataC.to(device)
                                target = target.to(device)
                                encoded_E = E_Supervised_Encoder(dataE)
                                encoded_M = M_Supervised_Encoder(dataM)
                                encoded_C = C_Supervised_Encoder(dataC)

                                intergrated_omics = torch.cat((encoded_E, encoded_M, encoded_C), 1)
                                Pred = Clas(intergrated_omics)

                                y_true = target.view(-1,1).cpu()

                                cl_loss = BCE_loss_fun(Pred,target.view(-1,1))
                                y_pred = Pred.cpu()
                                AUC = roc_auc_score(y_true.detach().numpy(),y_pred.detach().numpy())


                                Cl_optimizer.zero_grad()
                                cl_loss.backward()
                                Cl_optimizer.step()

                                epoch_cost = epoch_cost + (cl_loss / num_minibatches)
                                epoch_auc_list.append(AUC)
                                flag =1

                        if flag == 1:
                            cost_tr.append(torch.mean(epoch_cost))
                            auc_tr.append(np.mean(epoch_auc_list))
                            total_train_auc.append(np.mean(epoch_auc_list))
                            print('Iter-{}; Total loss: {:.4}'.format(cl_epoch, cl_loss))


                        with torch.no_grad():
                            Clas.eval()
                            encoded_val_E = E_Supervised_Encoder(TX_valE)
                            encoded_val_M = M_Supervised_Encoder(TX_valM)
                            encoded_val_C = C_Supervised_Encoder(TX_valC)

                            intergrated_val_omics = torch.cat((encoded_val_E, encoded_val_M, encoded_val_C), 1)
                            val_Pred = Clas(intergrated_val_omics)
                            val_loss = BCE_loss_fun(val_Pred,TY_val.view(-1,1))

                            val_y_true = TY_val.view(-1,1).cpu()
                            val_y_pred = val_Pred.cpu()

                            val_AUC = roc_auc_score(val_y_true.detach().numpy(),val_y_pred.detach().numpy())

                            print("val_AUC: ",val_AUC)

                            if pre_auc >= val_AUC:
                                break
                            else:
                                pre_auc = val_AUC

                    cost_val.append(val_loss)
                    auc_val.append(val_AUC)
                    total_val_auc.append(val_AUC)

                    Clas.eval()
                    encoded_test_E = E_Supervised_Encoder(TX_testE)
                    encoded_test_M = M_Supervised_Encoder(TX_testM)
                    encoded_test_C = C_Supervised_Encoder(TX_testC)

                    intergrated_test_omics = torch.cat((encoded_test_E, encoded_test_M, encoded_test_C), 1)
                    test_Pred = Clas(intergrated_test_omics)

                    test_y_true = TY_test.view(-1,1).cpu()
                    test_y_pred = test_Pred.cpu()

                    test_AUC = roc_auc_score(test_y_true.detach().numpy(),test_y_pred.detach().numpy())
                    total_test_auc.append(test_AUC)
                    title = str(datetime.now())+'iters {} epoch[E_Supervised_Encoder_epoch,M_Supervised_Encoder_epoch,C_Supervised_Encoder_epoch,Classifier_epoch] = ({},{},{},{}), mb_size = {},  out_dim[1,2,3] = ({},{},{}), marg = {}, lr[E,M,C] = ({}, {}, {}), Cwd = {},Ewd = {}, lrCL = {}, dropout[Supervised_Encoder,classifier]=({},{})'.\
                                  format(iters,E_Supervised_Encoder_epoch,C_Supervised_Encoder_epoch,M_Supervised_Encoder_epoch,Classifier_epoch, mb_size, OE_dim, OM_dim, OC_dim, marg, lrE, lrM, lrC, Cwd,Ewd, lrCL,E_dr,C_dr)

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

save_results_to_list = []
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set1/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set2/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set3/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set4/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set5/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set6/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set7/')
save_results_to_list.append('/NAS_Storage1/leo8544/SuperFELT_output/SuperFELT_output_set8/')
num_of_gpu = 7
gpu_num = 6
data_dir = '/GDSC/'
save_results = '/Super_FELT_GDSC_results/'

for i in range(len(hyperparameters_set_list)):
    work_num = gpu_num
    start = 0
    end = len(drugs)
    main(start, end, gpu_num,drugs,save_results,work_num,hyperparameters_set_list[i])
