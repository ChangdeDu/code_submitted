import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from scipy.io import loadmat
import numpy as np
from scipy import io

def fdr_correct_p(var):
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p

for subj in range(8):
    subject = 'subj0'+str(subj+1)
    mat_file_path = 'data/ROIs/rsa_scores_Subject_'+str(subj+1)+'_MLLM_bootstrap.mat'
    mat_contents = loadmat(mat_file_path)
    test_data=mat_contents['model_data']

    pcc_dist = list()
    repeat_num = test_data.shape[3]
    for rep in range(repeat_num):
        pcc = test_data[:,:,:,rep]
        a, b, c = pcc.shape
        flattened_pcc = pcc.ravel()
        pcc_dist.append(np.squeeze(flattened_pcc))

    pcc_dist = np.array(pcc_dist)
    fdr_p = fdr_correct_p(pcc_dist)
    print(np.sum(fdr_p[1] < 0.05))
    fdr_p = fdr_p[1]
    restored_array = fdr_p.reshape(a, b, c)
    fdr_p_dict = {'data': restored_array}
    io.savemat('data/ROIs/rsa_scores_Subject_'+str(subj+1)+'_MLLM_pvalue.mat', fdr_p_dict)