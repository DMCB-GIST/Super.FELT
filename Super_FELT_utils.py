import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def read_files(data_dir,GDSC_exprs,GDCS_mu, GDSC_cn, GDSC_y, External_exprs, External_mu, External_cn,External_y):
    ExternalE = pd.read_csv(data_dir+External_exprs,
                       sep = "\t", index_col=0, decimal = ",")
    ExternalE = pd.DataFrame.transpose(ExternalE)

    ExternalM = pd.read_csv(data_dir+External_mu,
                       sep = "\t", index_col=0, decimal = ".")
    ExternalM = pd.DataFrame.transpose(ExternalM)

    ExternalC = pd.read_csv(data_dir+External_cn,
                       sep = "\t", index_col=0, decimal = ".")
    ExternalC.drop_duplicates(keep='last')
    ExternalC = pd.DataFrame.transpose(ExternalC)
    ExternalC = ExternalC.loc[:,~ExternalC.columns.duplicated()]

    ExternalY = pd.read_csv(data_dir+External_y,
                           sep = "\t", index_col=0, decimal = ",")

    GDSCE = pd.read_csv(GDSC_exprs,
                        sep = "\t", index_col=0, decimal = ",")
    GDSCE = pd.DataFrame.transpose(GDSCE)

    GDSCM = pd.read_csv(GDCS_mu,
                        sep = "\t", index_col=0, decimal = ".")
    GDSCM = pd.DataFrame.transpose(GDSCM)


    GDSCC = pd.read_csv(GDSC_cn,
                        sep = "\t", index_col=0, decimal = ".")
    GDSCC.drop_duplicates(keep='last')
    GDSCC = pd.DataFrame.transpose(GDSCC)

    GDSCR = pd.read_csv(GDSC_y,
                        sep = "\t", index_col=0, decimal = ",")

    return GDSCE, GDSCM, GDSCC, GDSCR, ExternalE, ExternalM, ExternalC, ExternalY


def read_files_only_expres(data_dir,GDSC_exprs, GDSC_y, External_exprs, External_y):
    ExternalE = pd.read_csv(data_dir+External_exprs,
                       sep = "\t", index_col=0, decimal = ",")
    ExternalE = pd.DataFrame.transpose(ExternalE)

    ExternalY = pd.read_csv(data_dir+External_y,
                           sep = "\t", index_col=0, decimal = ",")

    GDSCE = pd.read_csv(GDSC_exprs,
                        sep = "\t", index_col=0, decimal = ",")
    GDSCE = pd.DataFrame.transpose(GDSCE)


    GDSCR = pd.read_csv(GDSC_y,
                        sep = "\t", index_col=0, decimal = ",")


    return GDSCE,GDSCR, ExternalE, ExternalY

def feature_selection(GDSCE, GDSCM, GDSCC):

    selector = VarianceThreshold(0.05*20)
    selector.fit_transform(GDSCE)
    GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

    selector = VarianceThreshold(0.00001*15)
    selector.fit_transform(GDSCM)
    GDSCM = GDSCM[GDSCM.columns[selector.get_support(indices=True)]]

    selector = VarianceThreshold(0.01*20)
    selector.fit_transform(GDSCC)
    GDSCC = GDSCC[GDSCC.columns[selector.get_support(indices=True)]]

    return GDSCE, GDSCM, GDSCC

def data_processing(GDSCE, GDSCM, GDSCC, GDSCR, ExternalE, ExternalM, ExternalC, ExternalY,External_data_name):

    GDSCM = GDSCM.fillna(0)
    GDSCM[GDSCM != 0.0] = 1
    GDSCC = GDSCC.fillna(0)
    GDSCC[GDSCC != 0.0] = 1

    ExternalC = ExternalC.fillna(0)
    ExternalC[ExternalC != 0.0] = 1
    ExternalM = ExternalM.fillna(0)
    ExternalM[ExternalM != 0.0] = 1
    GDSCR = GDSCR.drop(['logIC50', 'drug',  'exprs' , 'CNA', 'mutations'],axis=1)

    if External_data_name =='PDX':
        ExternalY = ExternalY.drop(['drug', 'ResponseCategory', 'Treatment', 'Treatment target',
               'Treatment type', 'BestResponse', 'Day_BestResponse', 'BestAvgResponse',
               'Day_BestAvgResponse', 'TimeToDouble', 'Day_Last', 'exprs', 'CNA',
               'mutations'],axis = 1)
    elif External_data_name =='TCGA':
        ExternalY = ExternalY.drop(['patient', 'cohort', 'drug', 'measure_of_response',
               'exprs_sample_barcode', 'cna_sample_barcode', 'mut_sample_barcode'],axis = 1)

    elif External_data_name == 'Sample':
        ExternalY = ExternalY.drop(['drug'],axis = 1)
        ExternalY.index = list(map(str, ExternalY.index.values))

    ls2 = set(GDSCE.index.values).intersection(set(GDSCM.index.values))
    ls2 = set(ls2).intersection(set(GDSCC.index.values))
    ls3 = set(ExternalE.index.values).intersection(set(ExternalM.index.values))
    ls3 = set(ls3).intersection(set(ExternalC.index.values))
    ls3 = set(ls3).intersection(set(ExternalY.index.values))
    if External_data_name == 'CCLE' or External_data_name == 'CTRP':
        non_common_genes = list(set(GDSCM.columns) - set(ExternalM.columns))
        new_df = pd.DataFrame(columns =non_common_genes)
        ExternalM = ExternalM.append(new_df).fillna(0)

    lsE = set(ExternalE.columns.values).intersection(set(GDSCE.columns.values))
    lsM = set(ExternalM.columns.values).intersection(set(GDSCM.columns.values))
    lsC = set(ExternalC.columns.values).intersection(set(GDSCC.columns.values))
    ExternalE = ExternalE.loc[ls3,lsE]
    ExternalM = ExternalM.loc[ls3,lsM]
    ExternalC = ExternalC.loc[ls3,lsC]
    GDSCE = GDSCE.loc[ls2,lsE]
    GDSCM = GDSCM.loc[ls2,lsM]
    GDSCC = GDSCC.loc[ls2,lsC]

    GDSCR.loc[GDSCR.iloc[:,0] == 'R'] = 0
    GDSCR.loc[GDSCR.iloc[:,0] == 'S'] = 1

    ls2 = list(map(int,ls2))
    GDSCR = GDSCR.loc[ls2,:]

    ExternalY = ExternalY.loc[ls3,:]
    ExternalY.loc[ExternalY.iloc[:,0] == 'R'] = 0
    ExternalY.loc[ExternalY.iloc[:,0] == 'S'] = 1


    ExternalE = ExternalE.apply(pd.to_numeric)
    ExternalM = ExternalM.apply(pd.to_numeric)
    ExternalC = ExternalC.apply(pd.to_numeric)
    GDSCE = GDSCE.apply(pd.to_numeric)
    GDSCM = GDSCM.apply(pd.to_numeric)
    GDSCC = GDSCC.apply(pd.to_numeric)

    return GDSCE, GDSCM, GDSCC, GDSCR, ExternalE, ExternalM, ExternalC, ExternalY

def data_processing_for_only_expres(GDSCE, GDSCR, ExternalE, ExternalY,External_data_name):

    GDSCR = GDSCR.drop(['logIC50', 'drug',  'exprs' , 'CNA', 'mutations'],axis=1)

    if External_data_name =='PDX':
        ExternalY = ExternalY.drop(['drug', 'ResponseCategory', 'Treatment', 'Treatment target',
               'Treatment type', 'BestResponse', 'Day_BestResponse', 'BestAvgResponse',
               'Day_BestAvgResponse', 'TimeToDouble', 'Day_Last', 'exprs', 'CNA',
               'mutations'],axis = 1)
    elif External_data_name =='TCGA':
        ExternalY = ExternalY.drop(['patient', 'cohort', 'drug', 'measure_of_response',
               'exprs_sample_barcode', 'cna_sample_barcode', 'mut_sample_barcode'],axis = 1)

    elif External_data_name == 'Sample':
        ExternalY = ExternalY.drop(['drug'],axis = 1)
        ExternalY.index = list(map(str, ExternalY.index.values))


    ls3 = set(ExternalE.index.values).intersection(set(ExternalY.index.values))

    lsE = set(ExternalE.columns.values).intersection(set(GDSCE.columns.values))

    ExternalE = ExternalE.loc[ls3,lsE]

    GDSCE = GDSCE.loc[:,lsE]

    GDSCR.loc[GDSCR.iloc[:,0] == 'R'] = 0
    GDSCR.loc[GDSCR.iloc[:,0] == 'S'] = 1

    ExternalY = ExternalY.loc[ls3,:]
    ExternalY.loc[ExternalY.iloc[:,0] == 'R'] = 0
    ExternalY.loc[ExternalY.iloc[:,0] == 'S'] = 1


    ExternalE = ExternalE.apply(pd.to_numeric)
    GDSCE = GDSCE.apply(pd.to_numeric)

    return GDSCE, GDSCR, ExternalE, ExternalY

def read_files_for_only_GDSC(data_dir,drug):
    try:
        exprs_file_name = "GDSC_exprs."+drug+".tsv"
        muation_file_name = "GDSC_mutations."+drug+".tsv"
        cn_file_name = "GDSC_CNA."+drug+".tsv"
        response_file_name = "GDSC_response."+drug+".tsv"
        GDSCE = pd.read_csv(data_dir+exprs_file_name,
                            sep = "\t", index_col=0, decimal = ",")
        GDSCE = pd.DataFrame.transpose(GDSCE)


        GDSCM = pd.read_csv(data_dir+muation_file_name,
                            sep = "\t", index_col=0, decimal = ".")
        GDSCM = pd.DataFrame.transpose(GDSCM)


        GDSCC = pd.read_csv(data_dir+cn_file_name,
                            sep = "\t", index_col=0, decimal = ".")

        GDSCC.drop_duplicates(keep='last')
        GDSCC = pd.DataFrame.transpose(GDSCC)

        GDSCR = pd.read_csv(data_dir+response_file_name,
                            sep = "\t", index_col=0, decimal = ",")

        GDSCR = GDSCR.drop(['logIC50', 'drug',  'exprs' , 'CNA', 'mutations'],axis=1)

        GDSCR.loc[GDSCR.iloc[:,0] == 'R'] = 0
        GDSCR.loc[GDSCR.iloc[:,0] == 'S'] = 1

        GDSCM = GDSCM.fillna(0)
        GDSCC = GDSCC.fillna(0)

        return GDSCE, GDSCM, GDSCC, GDSCR
    except:
        print('error')
        return [],[],[],[]

def processing_files_for_only_GDSC(GDSCE, GDSCM, GDSCC, GDSCR):

        ls2 = set(GDSCE.index.values).intersection(set(GDSCM.index.values))
        ls2 = set(ls2).intersection(set(GDSCC.index.values))
        selector = VarianceThreshold(0.05*20)
        selector.fit_transform(GDSCE)
        GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

        selector = VarianceThreshold(0.00001*15)
        selector.fit_transform(GDSCM)
        GDSCM = GDSCM[GDSCM.columns[selector.get_support(indices=True)]]

        selector = VarianceThreshold(0.01*20)
        selector.fit_transform(GDSCC)
        GDSCC = GDSCC[GDSCC.columns[selector.get_support(indices=True)]]

        GDSCE = GDSCE.loc[ls2,:]
        GDSCM = GDSCM.loc[ls2,:]
        GDSCC = GDSCC.loc[ls2,:]
        ls2 = list(map(int,ls2))
        GDSCR = GDSCR.loc[ls2,:]

        GDSCM[GDSCM != 0.0] = 1
        GDSCC[GDSCC != 0.0] = 1

        return GDSCE, GDSCM, GDSCC, GDSCR

def read_files_for_only_exprs_GDSC(data_dir,drug):
    try:
        exprs_file_name = "GDSC_exprs."+drug+".tsv"
        response_file_name = "GDSC_response."+drug+".tsv"
        GDSCE = pd.read_csv(data_dir+exprs_file_name,
                            sep = "\t", index_col=0, decimal = ",")
        GDSCE = pd.DataFrame.transpose(GDSCE)

        GDSCR = pd.read_csv(data_dir+response_file_name,
                            sep = "\t", index_col=0, decimal = ",")

        GDSCR = GDSCR.drop(['logIC50', 'drug',  'exprs' , 'CNA', 'mutations'],axis=1)

        GDSCR.loc[GDSCR.iloc[:,0] == 'R'] = 0
        GDSCR.loc[GDSCR.iloc[:,0] == 'S'] = 1

        return GDSCE, GDSCR
    except:
        print('error')
        return [],[],[],[]

def processing_files_for_only_exprs_GDSC(GDSCE, GDSCR):
        selector = VarianceThreshold(0.05*20)
        selector.fit_transform(GDSCE)
        GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

        return GDSCE, GDSCR
