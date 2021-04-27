import torch
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from Super_FELT_utils import read_files_for_only_GDSC


data_dir = '/GDSC/'
torch.manual_seed(42)

drugs = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])

def work(start,end,drugs,save_results_to):

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle = True)
    total_record_list = []
    for i in range(start,end):
        drug = drugs[i]

        origin_GDSCE,origin_GDSCM,origin_GDSCC,origin_GDSCR = read_files_for_only_GDSC(data_dir,drug)
        print("origin_GDSCE",drug)

        if len(origin_GDSCE) != 0:
            record_list = []
            total_test_auc = []
            total_val_auc = []

            AutoBorutaRF_genes = pd.read_csv('/AutoBorutaRF_genes.csv')
            exprs_genes = AutoBorutaRF_genes[AutoBorutaRF_genes['omics']=='rna']['ENTREZID']
            exprs_genes = [int(i) for i in exprs_genes]


            cna_genes = AutoBorutaRF_genes[AutoBorutaRF_genes['omics']=='cnv']['ENTREZID']
            cna_genes = [int(i) for i in cna_genes]

            common_exprs_genes = set(origin_GDSCE.columns).intersection(exprs_genes)
            common_cn_genes = set(origin_GDSCC.columns).intersection(cna_genes)


            GDSCE = origin_GDSCE[common_exprs_genes]
            GDSCC = origin_GDSCC[common_cn_genes]

            GDSCC = GDSCC.fillna(0)
            GDSCC[GDSCC != 0.0] = 1
            GDSCR = origin_GDSCR


            max_iter = 5
            for iters in range(max_iter):
                k = 0
                GDSCE,GDSCC,GDSCR=shuffle(GDSCE,GDSCC,GDSCR)
                Y = GDSCR['response'].values
                Y = sk.LabelEncoder().fit_transform(Y)

                for train_index, test_index in skf.split(GDSCE.values, Y):
                    k = k + 1

                    X_trainE = GDSCE.values[train_index,:]
                    X_testE =  GDSCE.values[test_index,:]
                    X_trainC = GDSCC.values[train_index,:]
                    X_testC = GDSCC.values[test_index,:]
                    Y_train = Y[train_index]
                    y_testE = Y[test_index]

                    scalerGDSC = sk.StandardScaler()
                    scalerGDSC.fit(X_trainE)
                    X_trainE = scalerGDSC.transform(X_trainE)
                    X_testE = scalerGDSC.transform(X_testE)

                    X_trainE, X_valE, X_trainC, X_valC, Y_train, Y_val \
                                = train_test_split(X_trainE, X_trainC, Y_train, test_size=0.2, random_state=42,stratify=Y_train)

                    intergrated_train_x = np.concatenate((X_trainE, X_trainC), 1)
                    intergrated_val_x = np.concatenate((X_valE, X_valC), 1)
                    intergrated_test_x = np.concatenate((X_testE, X_testC), 1)

                    R_index = [i for i in train_index if Y[i] == 0]
                    S_index = [i for i in train_index if Y[i] == 1]
                    N_S = len(S_index)
                    N_R = len(R_index)
                    T = N_R//N_S

                    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
                    clf.fit(intergrated_train_x, Y_train)
                    val_AUC = roc_auc_score(Y_val, clf.predict_proba(intergrated_val_x)[:, 1])

                    best_clf = clf
                    best_auc = val_AUC
                    for i in range(T):
                        clf = RandomForestClassifier(n_estimators=1000, random_state=0)
                        index = None

                        if i != T-1:
                            index = R_index[N_S*i:N_S*(i+1)]

                        else:
                            index = R_index[N_S*i:]

                        X_trainE = np.concatenate((GDSCE.values[index],
                                                   GDSCE.values[S_index]), 0)
                        X_trainC = np.concatenate((GDSCC.values[index],
                                                   GDSCC.values[S_index]), 0)
                        Y_train = np.concatenate((Y[index],
                                                  Y[S_index]), 0)


                        intergrated_train_x = np.concatenate((X_trainE,
                                                              X_trainC), 1)

                        clf.fit(intergrated_train_x, Y_train)

                        val_AUC = roc_auc_score(Y_val, clf.predict_proba(intergrated_val_x)[:, 1])

                        if val_AUC > best_auc:
                            best_clf = clf
                            best_auc = val_AUC

                    test_AUC = roc_auc_score(y_testE, best_clf.predict_proba(intergrated_test_x)[:, 1])

                    val_AUC = best_auc
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


save_results_to = '/AutoBorutaRF_GDSC'

start = 0
end = len(drugs)
#save_results_to = '/NAS_Storage1/leo8544/SuperFELT_output/RF_GDSC_mix1/'
work(start, end,drugs,save_results_to)
