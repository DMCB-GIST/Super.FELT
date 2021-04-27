import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from datetime import datetime

from utils import AllTripletSelector,AllPositivePairSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from models import Supervised_Encoder, Classifier, OnlineTriplet, OnlineTestTriplet
from Super_FELT_utils import read_files_only_expres, data_processing_for_only_expres
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold

def main(gpu_num,set_id,drug, External_data_name,hyperparameters_set,data_dir,save_results,
         GDSC_exprs_file,GDSC_y_file,External_exprs_file,External_y_file):
    #common hyperparameters
    marg = 1
    lrE = 0.01
    lrCL = 0.01

    mb_size = 55
    OE_dim = 256
    ICP_dim = OE_dim
    OCP_dim = ICP_dim


    #non-common hyperparameters
    E_Supervised_Encoder_epoch = hyperparameters_set['E_epoch']
    Classifier_epoch = hyperparameters_set['CL_epoch']

    E_dr = hyperparameters_set['E_dr']
    C_dr = hyperparameters_set['C_dr']
    Cwd = hyperparameters_set['Cwd']
    Ewd = hyperparameters_set['Ewd']

    torch.cuda.set_device(gpu_num)
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    #triplet_selector = RandomNegativeTripletSelector(marg)
    triplet_selector2 = AllTripletSelector()
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)
    BCE_loss_fun = torch.nn.BCELoss()

    skf = StratifiedKFold(n_splits=5, random_state=42)

    record_list = []
    total_train_auc = []
    total_test_auc = []
    total_External_auc = []
    GDSCE, GDSCR, ExternalE, ExternalY = read_files_only_expres(
                                data_dir,GDSC_exprs_file, GDSC_y_file,
                                External_exprs_file, External_y_file)

    selector = VarianceThreshold(0.05*20)
    selector.fit_transform(GDSCE)
    GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

    GDSCE, GDSCR, ExternalE, ExternalY = data_processing_for_only_expres(GDSCE, GDSCR,ExternalE,
                                                            ExternalY,External_data_name)

    best_encoded_multi_omics = None
    best_auc = 0

    max_iter = 5
    for iters in range(max_iter):
        k = 0
        GDSCE,GDSCR=shuffle(GDSCE,GDSCR)
        Y = GDSCR['response'].values
        Y = sk.LabelEncoder().fit_transform(Y)
        External_Y = ExternalY['response'].values
        External_Y = sk.LabelEncoder().fit_transform(External_Y)

        torch.cuda.empty_cache()
        for train_index, test_index in skf.split(GDSCE.values, Y):
            k = k + 1

            X_trainE = GDSCE.values[train_index,:]
            X_testE =  GDSCE.values[test_index,:]
            Y_train = Y[train_index]
            y_testE = Y[test_index]

            scalerGDSC = sk.StandardScaler()
            scalerGDSC.fit(X_trainE)
            X_trainE = scalerGDSC.transform(X_trainE)
            X_testE = scalerGDSC.transform(X_testE)


            TX_testE = torch.FloatTensor(X_testE).to(device)
            ty_testE = torch.FloatTensor(y_testE.astype(int)).to(device)

            tf_ExternalE = ExternalE.values

            tf_ExternalE = scalerGDSC.transform(tf_ExternalE)

            tf_ExternalE = torch.FloatTensor(tf_ExternalE).to(device)
            tf_ExternalY = torch.FloatTensor(External_Y.astype(int)).to(device)

            class_sample_count = np.array([len(np.where(Y_train==t)[0]) for t in np.unique(Y_train)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in Y_train])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

            trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE),
                                                          torch.FloatTensor(Y_train.astype(int)))

            trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler = sampler)

            n_sampE, IE_dim = X_trainE.shape

            cost_tr = []
            auc_tr = []
            cost_test = []
            auc_test = []

            torch.cuda.manual_seed_all(42)

            E_Supervised_Encoder = Supervised_Encoder(IE_dim,OE_dim,E_dr)

            E_Supervised_Encoder.to(device)

            E_optimizer = optim.Adagrad(E_Supervised_Encoder.parameters(), lr=lrE, weight_decay = Ewd)

            #TripSel2 = OnlineTriplet(marg, triplet_selector)
            TripSel = OnlineTestTriplet(marg, triplet_selector2)

            Clas = Classifier(OCP_dim,1,C_dr)
            Clas.to(device)
            Cl_optimizer = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = Cwd)


            ## train each Supervised_Encoder with thriplet loss
            break_num = 0
            pre_loss =100
            for e_epoch in range(E_Supervised_Encoder_epoch):
                E_Supervised_Encoder.train()
                flag = 0
                for i, (dataE,  target) in enumerate(trainLoader):
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
                    """
                        internal test
                    """
                    encoded_test_E = E_Supervised_Encoder(TX_testE)
                    E_Triplets_list = TripSel(encoded_test_E, ty_testE)
                    test_E_loss = trip_loss_fun(encoded_test_E[E_Triplets_list[:,0],:],encoded_test_E[E_Triplets_list[:,1],:],encoded_test_E[E_Triplets_list[:,2],:])

                    print("test_E_loss: ", test_E_loss)
                    if pre_loss <= test_E_loss:
                        break_num +=1

                    if break_num >1:
                        break
                    else:
                        pre_loss = test_E_loss

            E_Supervised_Encoder.eval()

            ## train classifier
            pre_auc = 0
            time = 0
            for cl_epoch in range(Classifier_epoch):
                epoch_cost = 0
                epoch_auc_list = []
                num_minibatches = int(n_sampE / mb_size)
                flag = 0
                Clas.train()
                for i, (dataE,  target) in enumerate(trainLoader):

                    if torch.mean(target)!=0. and torch.mean(target)!=1.:
                        dataE = dataE.to(device)
                        target = target.to(device)
                        encoded_E = E_Supervised_Encoder(dataE)

                        Pred = Clas(encoded_E)

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
                    """
                        validation
                    """
                    encoded_test_E = E_Supervised_Encoder(TX_testE)
                    test_Pred = Clas(encoded_test_E)
                    test_loss = BCE_loss_fun(test_Pred,ty_testE.view(-1,1))

                    test_y_true = ty_testE.view(-1,1).cpu()
                    test_y_pred = test_Pred.cpu()

                    test_AUC = roc_auc_score(test_y_true.detach().numpy(),test_y_pred.detach().numpy())

                    print("test_AUC",test_AUC)
                    if pre_auc >= test_AUC:
                        time +=1
                        if time >2:
                            break

                    else:
                        pre_auc = test_AUC

            Clas.eval()

            cost_test.append(test_loss)
            auc_test.append(test_AUC)
            total_test_auc.append(test_AUC)
            """
                External test
            """
            encoded_External_E = E_Supervised_Encoder(tf_ExternalE)

            External_Y_pred = Clas(encoded_External_E)

            External_Y_true = tf_ExternalY.view(-1,1).cpu()
            External_Y_pred = External_Y_pred.cpu()

            External_AUC = roc_auc_score(External_Y_true.detach().numpy(),External_Y_pred.detach().numpy())

            total_External_auc.append(External_AUC)

            print("################## drug ",drug)
            print("####################sum(auc_tr)/len(auc_tr): ", sum(auc_tr)/len(auc_tr))
            print("####################sum(auc_test)/len(auc_test): ", sum(auc_test)/len(auc_test))
            print("####################External_AUC: ", External_AUC)
            record_list.append([iters,sum(auc_tr)/len(auc_tr),sum(auc_test)/len(auc_test),External_AUC])

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_train_auc: ", sum(total_train_auc)/len(total_train_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@auc_External: ", sum(total_External_auc)/len(total_External_auc))
            print()


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_train_auc: ", sum(total_train_auc)/len(total_train_auc))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_auc_External: ", sum(total_External_auc)/len(total_External_auc))
    record_list.append(['total',np.average(total_train_auc),np.average(total_test_auc),np.average(total_External_auc)])
    record_df = pd.DataFrame(data = record_list,columns = ['iters','avg(auc_train)','avg(aucTest)','avg(auc_External)'])

    record_df.to_csv(save_results+'_'+str(datetime.now())+'result.txt',sep='\t',index=None)
    return np.average(total_train_auc),np.average(total_test_auc),np.average(total_External_auc)


drug_list = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])
External_data_name_list = ['TCGA','PDX','CCLE','CTRP']


hyperparameters_set_list = []
hyperparameters_set1 = {'E_dr':0.1, 'C_dr':0.1,'Cwd':0.0,'Ewd':0.0, 'E_epoch':10 ,'CL_epoch':5}
hyperparameters_set2 = {'E_dr':0.3, 'C_dr':0.3,'Cwd':0.01,'Ewd':0.01,'E_epoch':15 ,'CL_epoch':10}
hyperparameters_set3 = {'E_dr':0.3, 'C_dr':0.3,'Cwd':0.01,'Ewd':0.05,'E_epoch':10 ,'CL_epoch':15}
hyperparameters_set4 = {'E_dr':0.5, 'C_dr':0.5,'Cwd':0.01,'Ewd':0.01,'E_epoch':5 ,'CL_epoch':5}
hyperparameters_set5 = {'E_dr':0.5, 'C_dr':0.7,'Cwd':0.15,'Ewd':0.1,'E_epoch':15 ,'CL_epoch':5}
hyperparameters_set6 = {'E_dr':0.3, 'C_dr':0.5,'Cwd':0.01,'Ewd':0.01,'E_epoch':10 ,'CL_epoch':5}
hyperparameters_set7 = {'E_dr':0.4, 'C_dr':0.4,'Cwd':0.01,'Ewd':0.01,'E_epoch':0 ,'CL_epoch':10}
hyperparameters_set8 = {'E_dr':0.5, 'C_dr':0.5,'Cwd':0.1,'Ewd':0.1,'E_epoch':10 ,'CL_epoch':5}

hyperparameters_set_list.append(hyperparameters_set1)
hyperparameters_set_list.append(hyperparameters_set2)
hyperparameters_set_list.append(hyperparameters_set3)
hyperparameters_set_list.append(hyperparameters_set4)
hyperparameters_set_list.append(hyperparameters_set5)
hyperparameters_set_list.append(hyperparameters_set6)
hyperparameters_set_list.append(hyperparameters_set7)
hyperparameters_set_list.append(hyperparameters_set8)

gpu_num = 5
for drug in drug_list:
    for External_data_name in External_data_name_list:
        data_dir = '/external_data/'+External_data_name+'/'
        save_results_dir = '/MOLI_results/'
        GDSC_exprs_file = "/external_data/"+External_data_name\
                        +"/GDSC_exprs."+drug+".eb_with."+External_data_name+"_exprs."+drug+".tsv"

        GDCS_mu_file = "/GDSC/GDSC_mutations."+drug+".tsv"
        GDSC_cn_file = "/GDSC/GDSC_CNA."+drug+".tsv"
        GDSC_y_file = "/GDSC/GDSC_response."+drug+".tsv"
        External_exprs_file = External_data_name+"_exprs."+drug+".eb_with.GDSC_exprs."+drug+".tsv"
        External_y_file = External_data_name+"_response."+drug+".tsv"
        set_list = []
        for i in range(len(hyperparameters_set_list)):
            save_results = save_results_dir+External_data_name+'_'+drug+'_set'+str(i+1)
            try:
                train_auc, test_auc, external_auc = main(gpu_num = gpu_num,
                     set_id = 'set'+str(i+1), drug=drug,
                     External_data_name = External_data_name,
                     hyperparameters_set = hyperparameters_set_list[i],
                     data_dir = data_dir, save_results = save_results,
                     GDSC_exprs_file = GDSC_exprs_file,
                     GDSC_y_file = GDSC_y_file,
                     External_exprs_file = External_exprs_file,
                     External_y_file =  External_y_file
                     )
                set_list.append(['set'+str(i+1),train_auc, test_auc, external_auc])
            except:
                print(drug," doesn't have ",External_data_name)
                break
        if len(set_list)!=0:
            record_df = pd.DataFrame(data = set_list,columns = ['set','avg(auc_train)','avg(aucTest)','avg(auc_External)'])
            record_df.to_csv(save_results_dir+External_data_name+'_'+drug+'_result_'+str(datetime.now())+'.txt',sep='\t',index=None)
