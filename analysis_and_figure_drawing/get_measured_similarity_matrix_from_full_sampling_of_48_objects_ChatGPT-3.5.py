import numpy as np
import scipy.io as sio

with open('data/LLMs/ChatGPT-3.5/triplet_dataset/full_sampling_for_48_objects_ChatGPT-3.5(without_overlaping_with_trainset).txt', 'r') as file:
    lines = file.readlines()

trial_count = np.zeros((1854, 1854), dtype=int)

for line in lines:
    img1, img2, _ = map(int, line.strip().split())

    trial_count[img1, img2] += 1
    trial_count[img2, img1] += 1

def check_combination_presence(num1, num2):
    count = 0
    for line in lines:
        numbers = list(map(int, line.strip().split()))
        if num1 in numbers and num2 in numbers:
            count+=1

    return count

similarity_matrix = np.zeros((1854, 1854), dtype=float)

for i in range(1854):
    for j in range(i+1, 1854):
        print(i,j)       
        combination_present_count = check_combination_presence(i, j)
        
        if combination_present_count > 0:
            numerator = trial_count[i, j]
            similarity_score = numerator / combination_present_count
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score

np.fill_diagonal(similarity_matrix, 1)

sio.savemat('data/LLMs/ChatGPT-3.5/RSM1854_get_from_full_sampling_of_48_objects_ChatGPT-3.5.mat', {'RSM1854_triplet': similarity_matrix})