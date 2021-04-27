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

from models import Classifier
from Super_FELT_utils import read_files, data_processing, feature_selection
from matplotlib import pyplot as plt

from sklearn.feature_selection import VarianceThreshold

def main(gpu_num,set_id,drug, External_data_name,hyperparameters_set,data_dir, save_results,
         GDSC_exprs_file, GDCS_mu_file, GDSC_cn_file, GDSC_y_file,
         External_exprs_file, External_mu_file, External_cn_file,External_y_file):

    lrCL = 0.01

    mb_size =  55
    C_dr = hyperparameters_set['C_dr']
    Cwd = hyperparameters_set['Cwd']
    Classifier_epoch = hyperparameters_set['Classifier_epoch']

    torch.cuda.set_device(gpu_num)
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    BCE_loss_fun = torch.nn.BCELoss()

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle = True)

    record_list = []
    total_train_auc = []
    total_test_auc = []
    total_External_auc = []

    GDSCE, GDSCM, GDSCC, GDSCR, ExternalE, ExternalM, ExternalC, ExternalY = read_files(data_dir,
                                                                GDSC_exprs_file,GDCS_mu_file, GDSC_cn_file,
                                                                GDSC_y_file, External_exprs_file, External_mu_file,
                                                                External_cn_file,External_y_file)
    GDSCE, GDSCM, GDSCC = feature_selection(GDSCE, GDSCM, GDSCC)

    GDSCE, GDSCM, GDSCC, GDSCR, ExternalE, ExternalM, ExternalC, ExternalY = data_processing(GDSCE,
                                                                GDSCM, GDSCC, GDSCR, ExternalE,
                                                                ExternalM, ExternalC, ExternalY,External_data_name)

    max_iter = 5
    for iters in range(max_iter):
        k = 0
        GDSCE,GDSCM,GDSCC,GDSCR=shuffle(GDSCE,GDSCM,GDSCC,GDSCR)
        Y = GDSCR['response'].values
        Y = sk.LabelEncoder().fit_transform(Y)
        External_Y = ExternalY['response'].values
        External_Y = sk.LabelEncoder().fit_transform(External_Y)


        torch.cuda.empty_cache()
        for train_index, test_index in skf.split(GDSCE.values, Y):
            k = k + 1

            X_trainE = GDSCE.values[train_index,:]
            X_testE =  GDSCE.values[test_index,:]
            X_trainM = GDSCM.values[train_index,:]
            X_testM = GDSCM.values[test_index,:]
            X_trainC = GDSCC.values[train_index,:]
            X_testC = GDSCC.values[test_index,:]
            Y_train = Y[train_index]
            y_testE = Y[test_index]

            scalerGDSC = sk.StandardScaler()
            scalerGDSC.fit(X_trainE)
            X_trainE = scalerGDSC.transform(X_trainE)
            X_testE = scalerGDSC.transform(X_testE)

            X_trainM = np.nan_to_num(X_trainM)
            X_trainC = np.nan_to_num(X_trainC)
            X_testM = np.nan_to_num(X_testM)
            X_testC = np.nan_to_num(X_testC)

            TX_testE = torch.FloatTensor(X_testE).to(device)
            TX_testM = torch.FloatTensor(X_testM).to(device)
            TX_testC = torch.FloatTensor(X_testC).to(device)
            ty_testE = torch.FloatTensor(y_testE.astype(int)).to(device)

            tf_ExternalE = ExternalE.values
            tf_ExternalM = ExternalM.values
            tf_ExternalC = ExternalC.values

            tf_ExternalE = scalerGDSC.transform(tf_ExternalE)

            tf_ExternalE = torch.FloatTensor(tf_ExternalE).to(device)
            tf_ExternalM = torch.FloatTensor(tf_ExternalM).to(device)
            tf_ExternalC = torch.FloatTensor(tf_ExternalC).to(device)
            tf_ExternalY = torch.FloatTensor(External_Y.astype(int)).to(device)

            class_sample_count = np.array([len(np.where(Y_train==t)[0]) for t in np.unique(Y_train)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in Y_train])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

            trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                             torch.FloatTensor(X_trainC), torch.FloatTensor(Y_train.astype(int)))

            trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler = sampler)

            n_sampE, IE_dim = X_trainE.shape
            n_sampM, IM_dim = X_trainM.shape
            n_sampC, IC_dim = X_trainC.shape
            ICL_dim = IE_dim+IM_dim+IC_dim

            OE_dim = IE_dim
            OM_dim = IM_dim
            OC_dim = IC_dim

            cost_tr = []
            auc_tr = []
            cost_test = []
            auc_test = []

            torch.cuda.manual_seed_all(42)

            Clas = Classifier(ICL_dim,1,C_dr)
            Clas.to(device)
            Cl_optimizer = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = Cwd)

            ## train classifier
            for cl_epoch in range(Classifier_epoch):
                epoch_cost = 0
                epoch_auc_list = []
                num_minibatches = int(n_sampE / mb_size)
                flag = 0
                Clas.train()

                for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                    if torch.mean(target)!=0. and torch.mean(target)!=1.:
                        target = target.to(device)

                        intergrated_omics = torch.cat((dataE, dataM, dataC), 1)
                        intergrated_omics = intergrated_omics.to(device)

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
                    """
                        validation
                    """
                    intergrated_test_omics = torch.cat((TX_testE, TX_testM, TX_testC), 1)
                    intergrated_test_omics = intergrated_test_omics.to(device)

                    test_Pred = Clas(intergrated_test_omics)
                    test_loss = BCE_loss_fun(test_Pred,ty_testE.view(-1,1))

                    test_y_true = ty_testE.view(-1,1).cpu()
                    test_y_pred = test_Pred.cpu()

                    test_AUC = roc_auc_score(test_y_true.detach().numpy(),test_y_pred.detach().numpy())
                    cost_test.append(test_loss)
                    auc_test.append(test_AUC)
                    total_test_auc.append(test_AUC)
                    print("test_AUC ",test_AUC)
                    print("test_loss ",test_loss)


            """
                External test
            """

            intergrated_External_omics = torch.cat((tf_ExternalE, tf_ExternalM, tf_ExternalC), 1)
            External_Y_pred = Clas(intergrated_External_omics)

            External_Y_true = tf_ExternalY.view(-1,1).cpu()
            External_Y_pred = External_Y_pred.cpu()

            External_AUC = roc_auc_score(External_Y_true.detach().numpy(),External_Y_pred.detach().numpy())

            total_External_auc.append(External_AUC)


            title = str(datetime.now())+'iters {} epoch[Classifier_epoch] = ({}), mb_size = {},  out_dim[1,2,3] = ({},{},{}), Cwd = {}, lrCL = {}, dropout[classifier]=({})'.\
                                  format(iters,Classifier_epoch, mb_size, OE_dim, OM_dim, OC_dim,  Cwd, lrCL,C_dr)

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ drug: ",drug)
            print(title)
            print("####################sum(auc_tr)/len(auc_tr): ", sum(auc_tr)/len(auc_tr))
            print("####################sum(auc_test)/len(auc_test): ", sum(auc_test)/len(auc_test))
            print("####################External_AUC: ", External_AUC)
            record_list.append([iters,sum(auc_tr)/len(auc_tr),sum(auc_test)/len(auc_test),External_AUC])

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_train_auc: ", sum(total_train_auc)/len(total_train_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@auc_External: ", sum(total_External_auc)/len(total_External_auc))

    record_list.append(['total',np.average(total_train_auc),np.average(total_test_auc),np.average(total_External_auc)])
    record_list.append(['mb_size','OE_dim','OM_dim','OC_dim'])
    record_list.append([mb_size,OE_dim,OM_dim,OC_dim])
    record_list.append([title,'','',''])
    record_df = pd.DataFrame(data = record_list,columns = ['iters','avg(auc_train)','avg(aucTest)','avg(auc_External)'])

    record_df.to_csv(save_results+str(datetime.now())+'result.txt',sep='\t',index=None)
    np.average(total_train_auc)
    return np.average(total_train_auc),np.average(total_test_auc),np.average(total_External_auc)


drug_list = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])

External_data_name_list = ['PDX','TCGA','CCLE','CTRP']

hyperparameters_set_list = []
hyperparameters_set1 = {'C_dr':0.1,'Cwd':0.0,'Classifier_epoch':10}
hyperparameters_set2 = {'C_dr':0.4,'Cwd':0.01,'Classifier_epoch':15}
hyperparameters_set3 = {'C_dr':0.4,'Cwd':0.01,'Classifier_epoch':20}
hyperparameters_set4 = {'C_dr':0.5,'Cwd':0.05,'Classifier_epoch':10}
hyperparameters_set5 = {'C_dr':0.3,'Cwd':0.05,'Classifier_epoch':15}
hyperparameters_set6 = {'C_dr':0.3,'Cwd':0.05,'Classifier_epoch':20}
hyperparameters_set7 = {'C_dr':0.7,'Cwd':0.15,'Classifier_epoch':15}
hyperparameters_set8 = {'C_dr':0.0,'Cwd':0.0,'Classifier_epoch':10}

hyperparameters_set_list.append(hyperparameters_set1)
hyperparameters_set_list.append(hyperparameters_set2)
hyperparameters_set_list.append(hyperparameters_set3)
hyperparameters_set_list.append(hyperparameters_set4)
hyperparameters_set_list.append(hyperparameters_set5)
hyperparameters_set_list.append(hyperparameters_set6)
hyperparameters_set_list.append(hyperparameters_set7)
hyperparameters_set_list.append(hyperparameters_set8)

gpu_num = 2
for drug in drug_list:
    for External_data_name in External_data_name_list:
        data_dir = '/external_data/'+External_data_name+'/'
        save_results_dir = '/ANNF_results/'
        GDSC_exprs_file = "/external_data/"+External_data_name\
                        +"/GDSC_exprs."+drug+".eb_with."+External_data_name+"_exprs."+drug+".tsv"

        GDCS_mu_file = "/GDSC/GDSC_mutations."+drug+".tsv"
        GDSC_cn_file = "/GDSC/GDSC_CNA."+drug+".tsv"
        GDSC_y_file = "/GDSC/GDSC_response."+drug+".tsv"
        External_exprs_file = External_data_name+"_exprs."+drug+".eb_with.GDSC_exprs."+drug+".tsv"
        External_mu_file = External_data_name+"_mutations."+drug+".tsv"
        External_cn_file = External_data_name+"_CNA."+drug+".tsv"
        External_y_file = External_data_name+"_response."+drug+".tsv"
        set_list = []
        for i in range(len(hyperparameters_set_list)):
            save_results = save_results_dir+drug+'_set'+str(i+1)
            try:
                train_auc, test_auc, external_auc = main(gpu_num = gpu_num,
                     set_id = 'set'+str(i+1), drug=drug,
                     External_data_name = External_data_name,
                     hyperparameters_set = hyperparameters_set_list[i],
                     data_dir = data_dir, save_results = save_results,
                     GDSC_exprs_file = GDSC_exprs_file,
                     GDCS_mu_file = GDCS_mu_file,
                     GDSC_cn_file = GDSC_cn_file,
                     GDSC_y_file = GDSC_y_file,
                     External_exprs_file = External_exprs_file,
                     External_mu_file = External_mu_file,
                     External_cn_file = External_cn_file,
                     External_y_file =  External_y_file
                     )
                set_list.append(['set'+str(i+1),train_auc, test_auc, external_auc])
            except:
                print(drug," doesn't have ",External_data_name)
                break
        if len(set_list)!=0:
            record_df = pd.DataFrame(data = set_list,columns = ['set','avg(auc_train)','avg(aucTest)','avg(auc_External)'])
            record_df.to_csv(save_results_dir+External_data_name+'_'+drug+'_result_'+str(datetime.now())+'.txt',sep='\t',index=None)
