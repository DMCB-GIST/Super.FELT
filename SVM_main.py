import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from datetime import datetime

from Super_FELT_utils import read_files_only_expres, data_processing_for_only_expres
from sklearn import svm

from sklearn.feature_selection import VarianceThreshold

def main(drug, External_data_name,data_dir, save_results,
         GDSC_exprs_file,  GDSC_y_file, External_exprs_file,External_y_file):

    skf = StratifiedKFold(n_splits=5, random_state=42,shuffle = True)

    record_list = []
    total_test_auc = []
    total_External_auc = []


    GDSCE, GDSCR, ExternalE, ExternalY = read_files_only_expres(data_dir,
                                                                GDSC_exprs_file, GDSC_y_file,
                                                                External_exprs_file,External_y_file)

    selector = VarianceThreshold(0.05*20)
    selector.fit_transform(GDSCE)
    GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

    GDSCE,GDSCR, ExternalE, ExternalY = data_processing_for_only_expres(GDSCE, GDSCR,
                                                                ExternalE,ExternalY,External_data_name)

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

            tf_ExternalE = ExternalE.values

            tf_ExternalE = scalerGDSC.transform(tf_ExternalE)

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

            auc_test = []

            test_Pred = clf.predict_proba(X_testE)[:,1]

            test_AUC = roc_auc_score(y_testE, test_Pred)

            auc_test.append(test_AUC)
            total_test_auc.append(test_AUC)
            print("test_AUC ",test_AUC)

            """
                External test
            """

            External_Y_pred = clf.predict_proba(tf_ExternalE)[:,1]

            External_AUC = roc_auc_score(External_Y,External_Y_pred)

            total_External_auc.append(External_AUC)



            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ drug: ",drug)
            print("####################sum(auc_test)/len(auc_test): ", sum(auc_test)/len(auc_test))
            print("####################External_AUC: ", External_AUC)
            record_list.append([iters,sum(auc_test)/len(auc_test),External_AUC])

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@total_test_auc: ", sum(total_test_auc)/len(total_test_auc))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@auc_External: ", sum(total_External_auc)/len(total_External_auc))

    record_list.append(['total',np.average(total_test_auc),np.average(total_External_auc)])
    record_df = pd.DataFrame(data = record_list,columns = ['iters','avg(aucTest)','avg(auc_External)'])

    record_df.to_csv(save_results+str(datetime.now())+'result.txt',sep='\t',index=None)
    return np.average(total_test_auc),np.average(total_External_auc)


drug_list = list(pd.read_csv('GDSC_drugs.csv',sep='\n')['drugs'])

External_data_name_list = ['TCGA','PDX','CCLE','CTRP']

for drug in drug_list:
    for External_data_name in External_data_name_list:
        data_dir = '/external_data/'+External_data_name+'/'
        save_results_dir = '/SVM_results/'
        GDSC_exprs_file = "/external_data/"+External_data_name\
                        +"/GDSC_exprs."+drug+".eb_with."+External_data_name+"_exprs."+drug+".tsv"

        GDCS_mu_file = "/GDSC/GDSC_mutations."+drug+".tsv"
        GDSC_cn_file = "/GDSC/GDSC_CNA."+drug+".tsv"
        GDSC_y_file = "/GDSC/GDSC_response."+drug+".tsv"
        External_exprs_file = External_data_name+"_exprs."+drug+".eb_with.GDSC_exprs."+drug+".tsv"
        External_mu_file = External_data_name+"_mutations."+drug+".tsv"
        External_cn_file = External_data_name+"_CNA."+drug+".tsv"
        External_y_file = External_data_name+"_response."+drug+".tsv"
        try:
            save_results = save_results_dir+External_data_name+'_'+drug
            test_auc, external_auc = main(drug=drug,
                         External_data_name = External_data_name,
                         data_dir = data_dir, save_results = save_results,
                         GDSC_exprs_file = GDSC_exprs_file,
                         GDSC_y_file = GDSC_y_file,
                         External_exprs_file = External_exprs_file,
                         External_y_file =  External_y_file)
        except:
            print(drug," doesn't have ",External_data_name)
