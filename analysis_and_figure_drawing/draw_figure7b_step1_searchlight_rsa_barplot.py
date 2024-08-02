import cortex
import numpy as np
from scipy.io import loadmat
import scipy.io as sio

np.random.seed(1234)
xfm = 'func1pt8_to_anat0pt8_autoFSbbr'


roi_list = ['EarlyVis','RSC', 'OPA', 'EBA', 'FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca']
model_list =['LLM','VLM','Human','CLIPvision','CLIPtext']
for sub in range(8):
    subject = 'subj0' + str(sub+1) + '_Leila_Wehbe'
    # Get the map of which voxels are inside of our ROI
    roi_masks = cortex.utils.get_roi_masks(subject, xfm,
                                            roi_list=roi_list,
                                            gm_sampler='cortical-conservative',
                                            # Select only voxels mostly within cortex
                                            split_lr=False,  # No separate left/right ROIs
                                            threshold=None,  # Leave roi mask values as probabilites / fractions
                                            return_dict=True
                                            )
    roi_scores = np.zeros((len(model_list), len(roi_list)))
    roi_idx = 0
    for roi in roi_list:
        model_idx = 0
        for model in model_list:
            mask = roi_masks[roi]
            mask[mask != 0] = 1
            mat_file_path = 'data/ROIs/rsa_scores_Subject_' + str(sub+1) + '_'+model+'.mat'

            mat_contents = loadmat(mat_file_path)
            test_data = mat_contents['model_data']
            test_data = np.transpose(test_data, (2, 1, 0))

            test_data_roi = test_data * mask
            non_zero_elements = test_data_roi[test_data_roi != 0]

            roi_scores[model_idx,roi_idx] = np.mean(non_zero_elements)
            model_idx = model_idx + 1
        roi_idx = roi_idx + 1
    savepath = 'data/ROIs/'
    sio.savemat(savepath + 'roi_rsa_scores_Subject_'+str(sub+1) +'.mat', {'data': roi_scores})


