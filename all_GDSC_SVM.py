import torch
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from datetime import datetime

from Super_FELT_utils import read_files_for_only_exprs_GDSC, get_drug_name_list

from sklearn import svm
from sklearn.feature_selection import VarianceThreshold

data_dir = '/GDSC/'
save_results = '/SVM_GDSC_results/'
torch.manual_seed(42)

drugs = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])

def work(start,end,drugs,save_results_to):

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle = True)
    total_record_list = []
    for i in range(start,end):
        drug = drugs[i]

        origin_GDSCE,origin_GDSCR = read_files_for_only_exprs_GDSC(data_dir,drug)
        
        if len(origin_GDSCE) != 0:
            record_list = []
            total_test_auc = []
            total_val_auc = []

            selector = VarianceThreshold(0.05*20)
            selector.fit_transform(origin_GDSCE)
            GDSCE = origin_GDSCE[origin_GDSCE.columns[selector.get_support(indices=True)]]
            GDSCR = origin_GDSCR

            max_iter = 5
            for iters in range(max_iter):
                k = 0
                GDSCE,GDSCR = shuffle(GDSCE,GDSCR)
                Y = GDSCR['response'].values
                Y = sk.LabelEncoder().fit_transform(Y)

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

                    X_trainE, X_valE, Y_train, Y_val \
                                = train_test_split(X_trainE, Y_train, test_size=0.2, random_state=42,stratify=Y_train)


                    R_index = [i for i in train_index if Y[i] == 0]
                    S_index = [i for i in train_index if Y[i] == 1]
                    N_S = len(S_index)
                    N_R = len(R_index)
                    T = N_R//N_S

                    for i in range(T):
                        X_trainE = np.concatenate((X_trainE,
                                                   GDSCE.values[S_index]), 0)
                        Y_train = np.concatenate((Y_train,
                                                  Y[S_index]), 0)

                    X_trainE,Y_train=shuffle(X_trainE,Y_train)

                    clf = svm.SVC(probability=True)
                    clf.fit(X_trainE, Y_train)


                    val_pred = clf.predict_proba(X_valE)[:,1]
                    val_AUC = roc_auc_score(Y_val, val_pred)

                    test_Pred = clf.predict_proba(X_testE)[:,1]
                    test_AUC = roc_auc_score(y_testE, test_Pred)


                    total_val_auc.append(val_AUC)
                    total_test_auc.append(test_AUC)


                    print("################################### drug, ",drug)

                    print("####################val_AUC: ", val_AUC)
                    print("####################test_AUC: ",test_AUC)

                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Avg_total_val_auc: ", sum(total_val_auc)/len(total_val_auc))

                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Avg_total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
                    record_list.append([iters,val_AUC,test_AUC])


            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_val_auc: ", sum(total_val_auc)/len(total_val_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
            record_list.append(['total',sum(total_val_auc)/len(total_val_auc),sum(total_test_auc)/len(total_test_auc)])
            record_df = pd.DataFrame(data = record_list,columns = ['iters('+drug+')','avg(validation)','avg(aucTest)'])
            record_df.to_csv(save_results_to+str(datetime.now())+'_'+drug+'_result.txt',sep='\t',index=None)


            total_record_list.append([drug, sum(total_val_auc)/len(total_val_auc),sum(total_test_auc)/len(total_test_auc)])

    df = pd.DataFrame(data = total_record_list, columns = ['drug','AVG val AUC','AVG test AUC'])
    df.to_csv(save_results_to+str(datetime.now())+'_All_result.txt',sep='\t',index=None)


start = 0
end = len(drugs)

work(start, end,drugs,save_results_to)
