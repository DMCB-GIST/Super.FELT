import numpy as np
import pandas as pd
from binarizing_ic50 import Binarizing_ic50

ic50_df = pd.read_csv('ctrp_ic50.csv')
ic50_df = np.log(ic50_df)
drug_list = ic50_df.columns
avg_ci = 0.5/2
sigma = avg_ci/1.64

save_dir = './CTRPv2_drug_distribution/'

ctrp_gdsc_drug = pd.read_csv('ctrp_gdsc_drug.csv')

b_list = []
for drug in ctrp_gdsc_drug['CTRP name']:
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
b_df.to_csv('CTRPv2_log_threshold.csv',sep=',',index=False)

b_df = pd.read_csv('CTRPv2_log_threshold.csv',index_col = 0)

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
ic50_drugs_cells_df.to_csv('ctrp_log_ic50_binary.csv')

ic50_drugs_cells_df = pd.read_csv('ctrp_log_ic50_binary.csv',index_col=0)

ctrp_gdsc_drugs = pd.read_csv('ctrp_gdsc_drug.csv',index_col = 0)
ctrp_gdsc_drug_dict = ctrp_gdsc_drugs.to_dict()['GDSC1000 name(s)']


ctrp_gdsc_drug_dict['SN-38'] = 'SN-38'

gdsc_ctrp_id_name = pd.read_csv('gdsc_ccle_id_name.csv',index_col = 1)
gdsc_ctrp_id_name = gdsc_ctrp_id_name[~gdsc_ctrp_id_name.index.duplicated(keep='first')]

ctrp_cell_name = list(ic50_drugs_cells_df.columns)
for name in gdsc_ctrp_id_name.index:
    try:
        index = ctrp_cell_name.index(name)
        gdsc_id = gdsc_ctrp_id_name.loc[name]['id']
        ctrp_cell_name[index] = gdsc_id
    except:
        pass

ic50_drugs_cells_df.columns = ctrp_cell_name

gdsc_ctrp_id_name = pd.read_csv('gdsc_ccle_id_name.csv',index_col = 0)
gdsc_ctrp_id_name = gdsc_ctrp_id_name[~gdsc_ctrp_id_name.index.duplicated(keep='first')]
gdsc_to_ctrp = gdsc_ctrp_id_name.to_dict()['name']

unique_ctrp = []
no_drugs = []
for ctrp_drug in ctrp_gdsc_drug_dict.keys():
    try:
        ic50 = ic50_drugs_cells_df.loc[ctrp_drug]
        gdsc_drug = ctrp_gdsc_drug_dict[ctrp_drug]

        file = '/GDSC_response.'+gdsc_drug+'.tsv'
        gdsc_df = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_cell_lines = list(gdsc_df.index)

        ctrp = ic50_drugs_cells_df.loc[ctrp_drug]
        ctrp_cell_lines = list(ctrp.index)

        unique_ctrp_cell_lines = list(set(ctrp_cell_lines)-set(gdsc_cell_lines))
        unique_ctrp.append(ic50_drugs_cells_df.loc[ctrp_drug][unique_ctrp_cell_lines])
    except:
        print('drug is ',ctrp_drug)
        no_drugs.append(ctrp_drug)


unique_ctrp_df = pd.DataFrame(data=unique_ctrp)

unique_cells = list(unique_ctrp_df.columns)

for i in range(len(unique_cells)):
    try:
        unique_cells[i] = gdsc_to_ctrp[unique_cells[i]]
    except:
        pass

unique_ctrp_df.columns = unique_cells


unique_ctrp_df.to_csv('only_ctrp_cell_lines_log_ic50_binary.csv')

unique_ctrp_df = pd.read_csv('only_ctrp_cell_lines_log_ic50_binary.csv',index_col=0)

ensg_entre = pd.read_csv('ensg_entre.csv',index_col=0).dropna()
gene_entre = pd.read_csv('gene_entre.csv',index_col=0).dropna()

ctrp_exp = pd.read_csv('ctrp_exp.csv',index_col =0)
ctrp_mut= pd.read_csv('ctrp_mut.csv',index_col =0)

ensg_entre_dict = ensg_entre.to_dict()['EntrezGeneId']
ctrp_exp_genes = list(ctrp_exp.columns)
for i in range(len(ctrp_exp_genes)):
    try:
        ctrp_exp_genes[i] = (int(ensg_entre_dict[ctrp_exp_genes[i]]))
    except:
        pass

ctrp_exp.columns = ctrp_exp_genes
ctrp_expT = ctrp_exp.T
ctrp_genes = set(list(ctrp_expT.index))

for ctrp_drug in ctrp_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ctrp_gdsc_drug_dict[ctrp_drug]
        file = '/GDSC_exprs.'+gdsc_drug+'.tsv'
        gdsc_exprs = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_genes = set(list(gdsc_exprs.index))

        common_genes = list(gdsc_genes.intersection(ctrp_genes))

        ctrp_drug_exp =ctrp_expT.loc[common_genes].T#.dropna()
        unique_ctrp = unique_ctrp_df.loc[ctrp_drug].dropna()
        unique_ctrp_drug_exp = ctrp_drug_exp.loc[unique_ctrp.index]#.dropna()
        non_nan_list =[]
        for i in unique_ctrp_drug_exp.index:
            if len(unique_ctrp_drug_exp.loc[i].dropna()) !=0:
                non_nan_list.append(unique_ctrp_drug_exp.loc[i])
        unique_ctrp_drug_exp = pd.DataFrame(data =non_nan_list)
        unique_ctrp_drug_exp.T.to_csv('/CTRP_expres.'+gdsc_drug+'.tsv',sep='\t')
    except:
        print('drug is ',ctrp_drug)

gene_entre_dict = gene_entre.to_dict()['EntrezGeneId']
ctrp_mut_col = list(ctrp_mut.columns)
for i in range(len(ctrp_mut_col)):
    try:
        ctrp_mut_col[i] = (int(gene_entre_dict[ctrp_mut_col[i]]))
    except:
        pass

ctrp_mut.columns = ctrp_mut_col
ctrp_mutT = ctrp_mut.T
ctrp_mut_genes = set(list(ctrp_mutT.index))

for ctrp_drug in ctrp_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ctrp_gdsc_drug_dict[ctrp_drug]
        file = '/GDSC_mutations.'+gdsc_drug+'.tsv'
        gdsc_mut = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_genes = set(list(gdsc_mut.index))

        common_genes = list(gdsc_genes.intersection(ctrp_mut_genes))

        ctrp_drug_mut =ctrp_mutT.loc[common_genes].T#.dropna()
        unique_ctrp = unique_ctrp_df.loc[ctrp_drug].dropna()
        unique_ctrp_drug_mut = ctrp_drug_mut.loc[unique_ctrp.index]#.dropna()
        non_nan_list =[]
        for i in unique_ctrp_drug_mut.index:
            if len(unique_ctrp_drug_mut.loc[i].dropna()) !=0:
                non_nan_list.append(unique_ctrp_drug_mut.loc[i])
        unique_ctrp_drug_mut = pd.DataFrame(data =non_nan_list)
        unique_ctrp_drug_mut.T.to_csv('/CTRP_mutations.'+gdsc_drug+'.tsv',sep='\t')
    except:
        print('drug is ',ctrp_drug)

ccle_cna = pd.read_csv("CCLE_id.Segment_Mean.CNA.tsv",
                 sep = "\t",index_col = 0)
ccle_cnaT =ccle_cna.T
ccle_cnv_genes = set(list(ccle_cna.index))

for ctrp_drug in ctrp_gdsc_drug_dict.keys():
    try:
        gdsc_drug = ctrp_gdsc_drug_dict[ctrp_drug]
        file = '/GDSC_CNA.'+gdsc_drug+'.tsv'
        gdsc_cnv = pd.read_csv(file,sep='\t',index_col = 0)
        gdsc_genes = set(list(gdsc_cnv.index))

        common_genes = list(gdsc_genes.intersection(ccle_cnv_genes))

        ccle_drug_cnv =ccle_cna.loc[common_genes].T
        unique_ctrp = unique_ctrp_df.loc[ctrp_drug].dropna()

        unique_ccle_drug_cnv = ccle_drug_cnv.loc[unique_ctrp.index]
        non_nan_list =[]
        for i in unique_ccle_drug_cnv.index:
            if len(unique_ccle_drug_cnv.loc[i].dropna()) !=0:
                non_nan_list.append(unique_ccle_drug_cnv.loc[i])
        unique_ccle_drug_cnv = pd.DataFrame(data =non_nan_list)
        unique_ccle_drug_cnv.T.to_csv('/CTRP_CNA.'+gdsc_drug+'.tsv',sep='\t')
    except:
        print('drug is ',ctrp_drug )
        pass


ic50_drugs_cells_df = pd.read_csv('only_ctrp_cell_lines_log_ic50_binary.csv',index_col=0)
result = []
for ctrp_drug in ctrp_gdsc_drug_dict.keys():
    try:
        ic50 = ic50_drugs_cells_df.loc[ctrp_drug]
        gdsc_drug = ctrp_gdsc_drug_dict[ctrp_drug]

        df_cnv = pd.read_csv('/CTRP_CNA.'+gdsc_drug+'.tsv',
                             sep='\t',index_col =0)
        df_mut = pd.read_csv('/CTRP_mutations.'+gdsc_drug+'.tsv',
                             sep='\t',index_col =0)
        df_exprs = pd.read_csv('/CTRP_expres.'+gdsc_drug+'.tsv',
                               sep='\t',index_col =0)

        cnv_cells = set(df_cnv.columns)
        mut_cells = set(df_mut.columns)
        expres_cells = set(df_exprs.T.columns)
        ic50_cells = set(ic50.index)
        possible_cells=list(ic50_cells.intersection(cnv_cells).intersection(mut_cells).intersection(expres_cells))


        possible_ic50 = ic50.loc[possible_cells]

        df_exprs = df_exprs.T
        df_exprs = df_exprs.loc[possible_cells]
        df_exprs = df_exprs.T

        df_cnv = df_cnv.T
        df_cnv = df_cnv.loc[possible_cells]
        df_cnv = df_cnv.T

        df_mut = df_mut.T
        df_mut = df_mut.loc[possible_cells]
        df_mut = df_mut.T
        possible_ic50.to_csv('/CTRP_response.'+gdsc_drug+'.tsv',sep='\t',header=['response'])
        df_exprs.to_csv('/CTRP_expres.'+gdsc_drug+'.tsv',sep='\t')
        df_mut.to_csv('/CTRP_mutations.'+gdsc_drug+'.tsv',sep='\t')
        df_cnv.to_csv('/CTRP_CNA.'+gdsc_drug+'.tsv',sep='\t')

        print(ctrp_drug)
        s_list =[i for i in possible_ic50 if i =='S']
        print(len(s_list),len(possible_ic50))
        result.append([ctrp_drug,len(s_list),len(possible_ic50)-len(s_list),len(possible_ic50),b_df.loc[ctrp_drug]['threshold']])

    except:
        print('drug is  ',ctrp_drug)


result_df = pd.DataFrame(data=result,columns = ['drug','Num S','Num R','Total Num','Threshold'])
result_df.to_csv('CTRP_log_result.csv',sep=',',index=False)

result_df = pd.read_csv('CTRP_log_result.csv',sep=',',index_col = 0)
