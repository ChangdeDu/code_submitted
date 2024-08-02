import seaborn as sns
import matplotlib as mpl
from scipy.io import loadmat
import matplotlib.pyplot as plt

for subj in range(8):
    subject = 'subj0'+str(subj+1)
    mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(subj+1)+'_MLLM.mat'
    mat_contents = loadmat(mat_file_path)
    test_data=mat_contents['model_data']
    flattened_pcc_vlm = test_data.ravel()

    mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(subj+1)+'_LLM.mat'
    mat_contents = loadmat(mat_file_path)
    test_data=mat_contents['model_data']
    flattened_pcc_llm = test_data.ravel()

    mat_file_path = '/data/ROIs/rsa_scores_Subject_'+str(subj+1)+'_Human.mat'
    mat_contents = loadmat(mat_file_path)
    test_data=mat_contents['model_data']
    flattened_pcc_human = test_data.ravel()

    plt.figure()
    x = [-0.05, 1]
    y = [-0.05, 1]
    w = [-0.05, 0.85]
    sns.lineplot(x=x, y=y, linewidth=3, color="red", label="human level")
    sns.lineplot( x=x, y=w, linewidth=3, color="orange",  linestyle="--", label="85% human level")

    plt.hist2d(
        flattened_pcc_human,
        flattened_pcc_llm,
        bins=100,
        norm=mpl.colors.LogNorm(),
        cmap="magma",
    )
    plt.colorbar()
    plt.xlabel("Human Performance $(r)$", fontsize=15)
    plt.ylabel("LLM Performance $(r)$", fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(False)
    plt.savefig('/data/home/cddu/pythonProject/code/VICE/ROIs/2dhist_llm_vs_human_'+str(subj+1)+'.pdf')

    plt.figure()
    x = [-0.05, 1]
    y = [-0.05, 1]
    w = [-0.05, 0.85]
    sns.lineplot(x=x, y=y, linewidth=3, color="red", label="human level")
    sns.lineplot( x=x, y=w, linewidth=3, color="orange",  linestyle="--", label="85% human level")

    plt.hist2d(
        flattened_pcc_human,
        flattened_pcc_vlm,
        bins=100,
        norm=mpl.colors.LogNorm(),
        cmap="magma",
    )
    plt.colorbar()
    plt.xlabel("Human Performance $(r)$", fontsize=15)
    plt.ylabel("MLLM Performance $(r)$", fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(False)
    plt.savefig('/data/home/cddu/pythonProject/code/VICE/ROIs/2dhist_mllm_vs_human_'+str(subj+1)+'.pdf')