import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from datetime import datetime

from models import Classifier
from Super_FELT_utils import read_files_for_only_GDSC, processing_files_for_only_GDSC

torch.manual_seed(42)

drugs = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])

def work(start,end,gpu_num,drugs,save_results_to,work_num,hyperparameters_set):
    marg = 1
    lrCL = 0.01


    mb_size = 55
    C_dr = hyperparameters_set['C_dr']
    Cwd = hyperparameters_set['Cwd']
    Classifier_epoch = hyperparameters_set['Classifier_epoch']

    BCE_loss_fun = torch.nn.BCELoss()

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle=True)
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
                Y = preprocessing.LabelEncoder().fit_transform(Y)

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

                    OE_dim = IE_dim
                    OM_dim = IM_dim
                    OC_dim = IC_dim
                    ICL_dim = IE_dim+IM_dim+IC_dim

                    cost_tr = []
                    auc_tr = []
                    cost_val = []
                    auc_val = []

                    torch.cuda.manual_seed_all(42)

                    Clas = Classifier(ICL_dim,1,C_dr)
                    Clas.to(device)
                    Cl_optimizer = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = Cwd)

                    ## train classifier
                    pre_auc = 0
                    break_num = 0
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

                                intergrated_omics = torch.cat((dataE, dataM, dataC), 1)
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

                            intergrated_val_omics = torch.cat((TX_valE, TX_valM, TX_valC), 1)
                            val_Pred = Clas(intergrated_val_omics)
                            val_loss = BCE_loss_fun(val_Pred,TY_val.view(-1,1))

                            val_y_true = TY_val.view(-1,1).cpu()
                            val_y_pred = val_Pred.cpu()

                            val_AUC = roc_auc_score(val_y_true.detach().numpy(),val_y_pred.detach().numpy())
                            print("val_AUC: ",val_AUC)
                            if pre_auc >= val_AUC:
                                break_num +=1

                            if break_num >1:
                                break
                            else:
                                pre_auc = val_AUC

                    cost_val.append(val_loss)
                    auc_val.append(val_AUC)
                    total_val_auc.append(val_AUC)

                    Clas.eval()
                    intergrated_test_omics = torch.cat((TX_testE, TX_testM, TX_testC), 1)
                    test_Pred = Clas(intergrated_test_omics)
                    test_loss = BCE_loss_fun(test_Pred,TY_test.view(-1,1))

                    test_y_true = TY_test.view(-1,1).cpu()
                    test_y_pred = test_Pred.cpu()

                    test_AUC = roc_auc_score(test_y_true.detach().numpy(),test_y_pred.detach().numpy())
                    total_test_auc.append(test_AUC)
                    title = str(datetime.now())+'iters {} epoch[Classifier_epoch] = ({}), mb_size = {},  out_dim[1,2,3] = ({},{},{}), marg = {}, Cwd = {}, lrCL = {}, dropout[classifier]=({})'.\
                                  format(iters,Classifier_epoch, mb_size, OE_dim, OM_dim, OC_dim, marg, Cwd, lrCL,C_dr)

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
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_auc_Outer: ", sum(total_test_auc)/len(total_test_auc))
            record_list.append(['total',sum(total_train_auc)/len(total_train_auc),sum(total_val_auc)/len(total_val_auc),sum(total_test_auc)/len(total_test_auc)])
            record_list.append(['mb_size','OE_dim','OM_dim','OC_dim'])
            record_list.append([mb_size,OE_dim,OM_dim,OC_dim])
            record_list.append([title,'','',''])
            record_df = pd.DataFrame(data = record_list,columns = ['iters('+drug+')','avg(aucTrain)','avg(validation)','avg(aucTest)'])
            record_df.to_csv(save_results_to+str(datetime.now())+'_'+drug+'result.txt',sep='\t',index=None)




hyperparameters_set_list = []
hyperparameters_set1 = {'C_dr':0.1,'Cwd':0.0,'Classifier_epoch':10}
hyperparameters_set2 = {'C_dr':0.4,'Cwd':0.01,'Classifier_epoch':15}
hyperparameters_set3 = {'C_dr':0.4,'Cwd':0.01,'Classifier_epoch':20}
hyperparameters_set4 = {'C_dr':0.5,'Cwd':0.05,'Classifier_epoch':10}
hyperparameters_set5 = {'C_dr':0.3,'Cwd':0.05,'Classifier_epoch':15}
hyperparameters_set6 = {'C_dr':0.3,'Cwd':0.05,'Classifier_epoch':20}
hyperparameters_set7 = {'C_dr':0.7,'Cwd':0.15,'Classifier_epoch':15}
hyperparameters_set8 = {'C_dr':0.,'Cwd':0.0,'Classifier_epoch':10}

hyperparameters_set_list.append(hyperparameters_set1)
hyperparameters_set_list.append(hyperparameters_set2)
hyperparameters_set_list.append(hyperparameters_set3)
hyperparameters_set_list.append(hyperparameters_set4)
hyperparameters_set_list.append(hyperparameters_set5)
hyperparameters_set_list.append(hyperparameters_set6)
hyperparameters_set_list.append(hyperparameters_set7)
hyperparameters_set_list.append(hyperparameters_set8)
data_dir = '/GDSC/'
save_results = '/ANNF_GDSC_results/'

gpu_num = 0
for i in range(6,len(hyperparameters_set_list)):
    work_num = gpu_num
    start = 0
    end = int(len(drugs))

    work(start, end, gpu_num,drugs,save_results_to_list[i],work_num,hyperparameters_set_list[i])
