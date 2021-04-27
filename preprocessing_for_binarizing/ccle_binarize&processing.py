import numpy as np
import pandas as pd
from binarizing_ic50 import Binarizing_ic50

ic50_df = pd.read_csv('ccle_ic50.csv',index_col = 0)
ic50_df = np.log(ic50_df)
drug_list = ic50_df.columns
avg_ci = 0.5/2
sigma = avg_ci/1.64

save_dir = './ccle_drug_distribution/'

ccle_gdsc_drug = pd.read_csv('ccle_gdsc_drug.csv')

b_list = []
for drug in ccle_gdsc_drug['CCLE name']:
    try:
        ic50_list = ic50_df[drug].dropna()
        binarizer = Binarizing_ic50(ic50_list=ic50_list)
        binarizer.data_processing(max_ic50 = 15, min_ic50 = -10)
        binarizer.upsampling(sigma=sigma,NumSample=100)
        binarizer.make_kernel()
        binarizer.get_threta()
        b = binarizer.get_binary_threshold(save_dir)
        b_list.append([drug,b])

    except:
        print("error drug is ",drug)
        print(ic50_list)

b_df = pd.DataFrame(data = b_list, columns = ['drug','threshold'])
b_df.to_csv('CCLE_log_threshold.csv',sep=',',index=False)

b_df = pd.read_csv('CCLE_log_threshold.csv',index_col = 0)

ic50_drugs_cells = []
for drug in b_df.index:
    ic50_list = ic50_df[drug].dropna()
    threshold = b_df.loc[drug]['threshold']
    for cell_name in ic50_list.index:
        if ic50_list[cell_name] < threshold:
            ic50_list[cell_name] = 'S'
        else:
            ic50_list[cell_name] = 'R'
    ic50_drugs_cells.append(ic50_list)

ic50_drugs_cells_df = pd.DataFrame(data= ic50_drugs_cells)
ic50_drugs_cells_df.to_csv('ccle_log_ic50_binary.csv')
ic50_drugs_cells_df = pd.read_csv('ccle_log_ic50_binary.csv',index_col=0)


gdsc_drug_list = pd.read_csv('gdsc_drug_list.csv')
gdsc_drug_set = set(list(gdsc_drug_list.T.iloc[0]))

ccle_gdsc_drugs = pd.read_csv('ccle_gdsc_drug.csv',index_col = 0)
ccle_gdsc_drug_list = list(ccle_gdsc_drugs['GDSC1000 name(s)'])
gdsc_drugs = list(set(ccle_gdsc_drug_list).intersection(gdsc_drug_set))

ccle_gdsc_drugs = pd.read_csv('ccle_gdsc_drug.csv',index_col = 1)
ccle_gdsc_drug_dict = ccle_gdsc_drugs.to_dict()['GDSC name']

gdsc_ccle_id_name = pd.read_csv('gdsc_ccle_id_name.csv',index_col = 1)
gdsc_ccle_id_name = gdsc_ccle_id_name[~gdsc_ccle_id_name.index.duplicated(keep='first')]

ccle_cell_name = list(ic50_drugs_cells_df.columns)
for name in gdsc_ccle_id_name.index:
    try:
        index = ccle_cell_name.index(name)
        gdsc_id = gdsc_ccle_id_name.loc[name]['id']
        ccle_cell_name[index] = gdsc_id
    except:
        pass

ic50_drugs_cells_df.columns = ccle_cell_name

gdsc_ccle_id_name = pd.read_csv('gdsc_ccle_id_name.csv',index_col = 0)
gdsc_ccle_id_name = gdsc_ccle_id_name[~gdsc_ccle_id_name.index.duplicated(keep='first')]
gdsc_to_ccle = gdsc_ccle_id_name.to_dict()['name']

unique_ccle = []
for ccle_drug in ccle_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ccle_gdsc_drug_dict[ccle_drug]

        file = '/GDSC_response.'+gdsc_drug+'.tsv'
        gdsc_df = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_cell_lines = list(gdsc_df.index)

        ccle = ic50_drugs_cells_df.loc[ccle_drug]
        ctrp_cell_lines = list(ccle.index)

        unique_ccle_cell_lines = list(set(ctrp_cell_lines)-set(gdsc_cell_lines))
        unique_ccle.append(ic50_drugs_cells_df.loc[ccle_drug][unique_ccle_cell_lines])
    except:
        print('drug ',ccle_drug)


unique_ccle_df = pd.DataFrame(data=unique_ccle)

unique_cells = list(unique_ccle_df.columns)

for i in range(len(unique_cells)):
    try:
        unique_cells[i] = unique_ccle_df[unique_cells[i]]
    except:
        pass
unique_ccle_df.columns = unique_cells


#unique_ctrp_df.to_csv('only_ccle_cell_lines_log_ic50_binary.csv')

unique_ccle_df = pd.read_csv('only_ccle_cell_lines_log_ic50_binary.csv',index_col=0)

ensg_entre = pd.read_csv('ensg_entre.csv',index_col=0).dropna()
gene_entre = pd.read_csv('gene_entre.csv',index_col=0).dropna()


ccle_exp = pd.read_csv('ccle_exp_orignal.csv',index_col =0)
ccle_mut= pd.read_csv('ccle_mut.csv',index_col =0)

gene_entre_dict = gene_entre.to_dict()['EntrezGeneId']
ensg_entre_dict = ensg_entre.to_dict()['EntrezGeneId']

ccle_exp_genes = list(ccle_exp.columns)
for i in range(len(ccle_exp_genes)):
    try:
        ccle_exp_genes[i] = (int(ensg_entre_dict[ccle_exp_genes[i]]))
    except:
        pass


ccle_exp.columns = ccle_exp_genes

ccle_expT = ccle_exp.T
ccle_genes = set(list(ccle_expT.index))

for ccle_drug in ccle_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ccle_gdsc_drug_dict[ccle_drug]
        file = '/GDSC_exprs.'+gdsc_drug+'.tsv'
        gdsc_exprs = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_genes = set(list(gdsc_exprs.index))

        common_genes = list(gdsc_genes.intersection(ccle_genes))

        ccle_drug_exp =ccle_expT.loc[common_genes].T#.dropna()
        unique_ccle = unique_ccle_df.loc[ccle_drug].dropna()
        unique_ctrp_drug_exp = ccle_drug_exp.loc[unique_ccle.index]#.dropna()
        non_nan_list =[]
        for i in unique_ctrp_drug_exp.index:
            if len(unique_ctrp_drug_exp.loc[i].dropna()) !=0:
                non_nan_list.append(unique_ctrp_drug_exp.loc[i])
        unique_ctrp_drug_exp = pd.DataFrame(data =non_nan_list)
        unique_ctrp_drug_exp.T.to_csv('/CCLE_expres.'+gdsc_drug+'.tsv',sep='\t')
    except:
        print('drug is ',ccle_drug)

gene_entre_dict = gene_entre.to_dict()['EntrezGeneId']
ccle_mut_genes = list(ccle_mut.columns)
for i in range(len(ccle_mut_genes)):
    try:
        ccle_mut_genes[i] = (int(gene_entre_dict[ccle_mut_genes[i]]))
    except:
        pass

ccle_mut.columns = ccle_mut_genes
ccle_mutT = ccle_mut.T
ccle_mut_genes = set(list(ccle_mutT.index))

for ccle_drug in ccle_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ccle_gdsc_drug_dict[ccle_drug]

        file = '/GDSC_mutations.'+gdsc_drug+'.tsv'
        gdsc_mut = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_genes = set(list(gdsc_mut.index))

        common_genes = list(gdsc_genes.intersection(ccle_mut_genes))

        ccle_drug_mut =ccle_mutT.loc[common_genes].T#.dropna()
        unique_ccle = unique_ccle_df.loc[ccle_drug].dropna()
        unique_ccle_drug_mut = ccle_drug_mut.loc[unique_ccle.index]#.dropna()
        non_nan_list =[]
        for i in unique_ccle_drug_mut.index:
            if len(unique_ccle_drug_mut.loc[i].dropna()) !=0:
                non_nan_list.append(unique_ccle_drug_mut.loc[i])
        unique_ccle_drug_mut = pd.DataFrame(data =non_nan_list)
        unique_ccle_drug_mut.T.to_csv('/CCLE_mutations.'+gdsc_drug+'.tsv',sep='\t')
    except:
        pass

ccle_cna = pd.read_csv("CCLE_id.Segment_Mean.CNA.tsv",
                 sep = "\t",index_col = 0)
ccle_cnaT =ccle_cna.T
ccle_cnv_genes = set(list(ccle_cna.index))

for ccle_drug in ccle_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ccle_gdsc_drug_dict[ccle_drug]
        file = '/GDSC_CNA.'+gdsc_drug+'.tsv'
        gdsc_cnv = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_genes = set(list(gdsc_cnv.index))

        common_genes = list(gdsc_genes.intersection(ccle_cnv_genes))

        ccle_drug_cnv =ccle_cna.loc[common_genes].T
        unique_ccle = unique_ccle_df.loc[ccle_drug].dropna()

        unique_ccle_drug_cnv = ccle_drug_cnv.loc[unique_ccle.index]
        non_nan_list =[]
        for i in unique_ccle_drug_cnv.index:
            if len(unique_ccle_drug_cnv.loc[i].dropna()) !=0:
                non_nan_list.append(unique_ccle_drug_cnv.loc[i])
        unique_ccle_drug_cnv = pd.DataFrame(data =non_nan_list)
        unique_ccle_drug_cnv.T.to_csv('/CCLE_CNA.'+gdsc_drug+'.tsv',sep='\t')
    except:
        print('drug is ',ccle_drug )
        pass

ic50_drugs_cells_df = pd.read_csv('only_ccle_cell_lines_log_ic50_binary.csv',index_col=0)
result = []
for ccle_drug in ccle_gdsc_drug_dict.keys():
    try:
        ic50 = ic50_drugs_cells_df.loc[ccle_drug]
        gdsc_drug = ccle_gdsc_drug_dict[ccle_drug]

        df_cnv = pd.read_csv('/CCLE_CNA.'+gdsc_drug+'.tsv',
                             sep='\t',index_col =0)
        df_mut = pd.read_csv('/CCLE_mutations.'+gdsc_drug+'.tsv',
                             sep='\t',index_col =0)
        df_exprs = pd.read_csv('/CCLE_expres.'+gdsc_drug+'.tsv',
                               sep='\t',index_col =0)

        cnv_cells = set(df_cnv.columns)
        mut_cells = set(df_mut.columns)
        expres_cells = set(df_exprs.columns)
        ic50_cells = set(ic50.index)
        possible_cells=list(ic50_cells.intersection(cnv_cells).intersection(mut_cells).intersection(expres_cells))
        possible_ic50 = ic50.loc[possible_cells]

        df_cnv = df_cnv.T
        df_cnv = df_cnv.loc[possible_cells]
        df_cnv = df_cnv.T

        df_mut = df_mut.T
        df_mut = df_mut.loc[possible_cells]
        df_mut = df_mut.T

        df_exprs = df_exprs.T
        df_exprs = df_exprs.loc[possible_cells]
        df_exprs = df_exprs.T

        possible_ic50.to_csv('/CCLE_response.'+gdsc_drug+'.tsv',sep='\t',header=['response'])
        df_exprs = pd.read_csv('/CCLE_expres.'+gdsc_drug+'.tsv',
                               sep='\t',index_col =0)
        df_mut.to_csv('/CCLE_mutations.'+gdsc_drug+'.tsv',sep='\t')
        df_cnv.to_csv('/CCLE_CNA.'+gdsc_drug+'.tsv',sep='\t')

        print(gdsc_drug)
        s_list =[i for i in possible_ic50 if i =='S']
        print(len(s_list),len(possible_ic50))
        result.append([gdsc_drug,len(s_list),len(possible_ic50)-len(s_list),len(possible_ic50),b_df.loc[ccle_drug]['threshold']])

    except:
        print('drug is  ' ,ccle_drug)

result_df = pd.DataFrame(data=result,columns = ['drug','Num S','Num R','Total Num','Threshold'])


result_df.to_csv('CCLE_log_result.csv',sep=',',index=False)

result_df = pd.read_csv('CCLE_log_result.csv',sep=',',index_col = 0)
