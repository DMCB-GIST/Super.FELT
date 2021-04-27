import torch
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from Super_FELT_utils import read_files

def main(drug, External_data_name,data_dir,save_results,
         GDSC_exprs_file, GDCS_mu_file, GDSC_cn_file, GDSC_y_file,
         External_exprs_file, External_mu_file, External_cn_file,External_y_file):

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle = True)
    record_list = []
    total_test_auc = []
    total_External_auc = []

    GDSCE, GDSCM, GDSCC, GDSCR, ExternalE, ExternalM, ExternalC, ExternalY = read_files(data_dir,
                                                GDSC_exprs_file,GDCS_mu_file, GDSC_cn_file, GDSC_y_file,
                                                External_exprs_file, External_mu_file, External_cn_file,External_y_file)

    AutoBorutaRF_genes = pd.read_csv('/AutoBorutaRF_genes.csv')
    exprs_genes = AutoBorutaRF_genes[AutoBorutaRF_genes['omics']=='rna']['ENTREZID']
    exprs_genes = [int(i) for i in exprs_genes]


    cna_genes = AutoBorutaRF_genes[AutoBorutaRF_genes['omics']=='cnv']['ENTREZID']
    cna_genes = [int(i) for i in cna_genes]

    common_exprs_genes = set(GDSCE.columns).intersection(exprs_genes).intersection(ExternalE.columns)
    common_cn_genes = set(GDSCC.columns).intersection(cna_genes).intersection(ExternalC.columns)


    GDSCE = GDSCE[common_exprs_genes]
    ExternalE = ExternalE[common_exprs_genes]

    GDSCC = GDSCC[common_cn_genes]
    ExternalC = ExternalC[common_cn_genes]

    GDSCC = GDSCC.fillna(0)
    GDSCC[GDSCC != 0.0] = 1

    ExternalC = ExternalC.fillna(0)
    ExternalC[ExternalC != 0.0] = 1

    ls3 = set(ExternalE.index.values).intersection(set(ExternalM.index.values))
    ls3 = set(ls3).intersection(set(ExternalC.index.values))
    ls3 = set(ls3).intersection(set(ExternalY.index.values))
    ExternalE = ExternalE.loc[ls3,:]
    ExternalM = ExternalM.loc[ls3,:]
    ExternalC = ExternalC.loc[ls3,:]




    max_iter = 5
    for iters in range(max_iter):
        k = 0
        GDSCE,GDSCC,GDSCR=shuffle(GDSCE,GDSCC,GDSCR)
        Y = GDSCR['response'].values
        Y = sk.LabelEncoder().fit_transform(Y)
        External_Y = ExternalY['response'].values
        External_Y = sk.LabelEncoder().fit_transform(External_Y)

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

            X_trainC = np.nan_to_num(X_trainC)
            X_testC = np.nan_to_num(X_testC)


            tf_ExternalE = ExternalE.values
            tf_ExternalC = ExternalC.values

            tf_ExternalE = scalerGDSC.transform(tf_ExternalE)


            R_index = [i for i in train_index if Y[i] == 0]
            S_index = [i for i in train_index if Y[i] == 1]
            N_S = len(S_index)
            N_R = len(R_index)
            T = N_R//N_S


            intergrated_train_x = np.concatenate((X_trainE, X_trainC), 1)
            intergrated_test_x = np.concatenate((X_testE, X_testC), 1)
            intergrated_external_x = np.concatenate((tf_ExternalE, tf_ExternalC), 1)

            clf = RandomForestClassifier(n_estimators=1000, random_state=0)
            clf.fit(intergrated_train_x, Y_train)
            test_AUC = roc_auc_score(y_testE, clf.predict_proba(intergrated_test_x)[:, 1])

            best_clf = clf
            best_auc = test_AUC

            for i in range(T):
                clf = RandomForestClassifier(n_estimators=1000, random_state=0)
                intergrated_train_x = None
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

                test_AUC = roc_auc_score(y_testE, clf.predict_proba(intergrated_test_x)[:, 1])

                if test_AUC > best_auc:
                    best_clf = clf
                    best_auc = test_AUC

            External_AUC = roc_auc_score(External_Y, best_clf.predict_proba(intergrated_external_x)[:, 1])

            test_AUC = best_auc
            total_test_auc.append(test_AUC)

            total_External_auc.append(External_AUC)

            print("################################### drug, ",drug)
            print("####################test_AUC: ",test_AUC)
            print("####################External_AUC: ", External_AUC)
            record_list.append([iters,test_AUC,External_AUC])

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Avg_total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Avg_External: ", sum(total_External_auc)/len(total_External_auc))


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_train_auc: ", sum(total_test_auc)/len(total_test_auc))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_auc_External: ", sum(total_External_auc)/len(total_External_auc))
    record_list.append(['AVG',np.average(total_test_auc),np.average(total_External_auc)])
    record_df = pd.DataFrame(data = record_list,columns = ['iters','avg(aucTest)','avg(auc_External)'])

    record_df.to_csv(save_results+'_'+str(datetime.now())+'_result.txt',sep='\t',index=None)


    return np.average(total_test_auc),np.average(total_External_auc)


drug_list = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])
External_data_name_list = ['TCGA','PDX','CCLE','CTRP']

for drug in drug_list:
    for External_data_name in External_data_name_list:
        data_dir = '/external_data/'+External_data_name+'/'
        save_results_dir = '/RF_results/'

        GDSC_exprs_file = "/external_data/"+External_data_name\
                        +"/GDSC_exprs."+drug+".eb_with."+External_data_name+"_exprs."+drug+".tsv"

        GDCS_mu_file = "/GDSC/GDSC_mutations."+drug+".tsv"
        GDSC_cn_file = "/GDSC/GDSC_CNA."+drug+".tsv"
        GDSC_y_file = "/GDSC/GDSC_response."+drug+".tsv"
        External_exprs_file = External_data_name+"_exprs."+drug+".eb_with.GDSC_exprs."+drug+".tsv"
        External_mu_file = External_data_name+"_mutations."+drug+".tsv"
        External_cn_file = External_data_name+"_CNA."+drug+".tsv"
        External_y_file = External_data_name+"_response."+drug+".tsv"
        save_results = save_results_dir+drug+'_'+External_data_name

        try:
            test_auc, external_auc = main(drug=drug,save_results = save_results,
                         External_data_name = External_data_name,
                         data_dir = data_dir,
                         GDSC_exprs_file = GDSC_exprs_file,
                         GDCS_mu_file = GDCS_mu_file,
                         GDSC_cn_file = GDSC_cn_file,
                         GDSC_y_file = GDSC_y_file,
                         External_exprs_file = External_exprs_file,
                         External_mu_file = External_mu_file,
                         External_cn_file = External_cn_file,
                         External_y_file =  External_y_file
                         )
            data.append([drug, External_data_name,test_auc, external_auc])

        except:
            print(drug," doesn't have ",External_data_name)


record_df = pd.DataFrame(data = data,columns = ['drug','External_data_name','avg(aucTest)','avg(auc_External)'])
record_df.to_csv(save_results_dir+'RF_external_result_'+str(datetime.now())+'.txt',sep='\t',index=None)
