import cortex
import numpy as np
from scipy.io import loadmat

np.random.seed(1234)

sub = 1 # 2,3,4,5,6,7,8
subject = 'subj0'+str(sub)
xfm = 'func1pt8_to_anat0pt8_autoFSbbr'

#### for MLLM
mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(sub)+'_MLLM.mat'
mat_contents = loadmat(mat_file_path)
test_data_MLLM=mat_contents['model_data']
test_data = np.transpose(test_data_MLLM, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_MLLM)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for LLM
mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(sub)+'_LLM.mat'
mat_contents = loadmat(mat_file_path)
test_data_LLM=mat_contents['model_data']
test_data = np.transpose(test_data_LLM, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_MLLM)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for Human
mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(sub)+'_Human.mat'
mat_contents = loadmat(mat_file_path)
test_data_Human=mat_contents['model_data']
test_data = np.transpose(test_data_Human, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_MLLM)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for CLIPvision
mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(sub)+'_CLIPvision.mat'
mat_contents = loadmat(mat_file_path)
test_data_CLIPvision=mat_contents['model_data']
test_data = np.transpose(test_data_CLIPvision, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_MLLM)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for CLIPtext
mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(sub)+'_CLIPtext.mat'
mat_contents = loadmat(mat_file_path)
test_data_CLIPtext=mat_contents['model_data']
test_data = np.transpose(test_data_CLIPtext, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_MLLM)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)