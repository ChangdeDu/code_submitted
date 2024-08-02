import numpy as np
from himalaya.ridge import RidgeCV

## using multimodal features as input
X_image = np.load('things1854image_feas_CLIP_ViT_L14.npy')
X_text = np.load('things1854image_caption_feas_CLIP_ViT_L14.npy')
X = np.concatenate((X_image, X_text), axis=1)
Y = np.load('data/MLLMs/Gemini_Pro_Vision/spose_embedding_66d_sorted_gemini.txt')

## using linear regression
model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 150, 200, 300, 500, 700, 1000])
model.fit(X, Y)
print(model.best_alphas_)
X_test_image = np.load('44image_feas_CLIP_ViT_L14.npy')
X_test_text = np.load('44image_caption_feas_CLIP_ViT_L14.npy')
X_test = np.concatenate((X_test_image, X_test_text), axis=1)
nsd44_predict = model.predict(X_test)
np.savetxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_nsd44_predicted_from_gemini.txt',
           nsd44_predict, fmt='%.8f')



