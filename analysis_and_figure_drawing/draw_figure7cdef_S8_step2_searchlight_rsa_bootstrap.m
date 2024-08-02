
% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes

addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

re_rankindex = 1+[0 6 8 11 16 21 35 37 4 25 24 34 40 22 10 1 2 15 27 28 30 43 17 26 7 9 13 20 19 18 41 14 29 42 3 5 23 31 32 12 36 38 39 33];

repeat = 100; % bootstrap repeats
label_idx = 1:length(re_rankindex);

subj_id = [1,2,3,4,5,6,7,8];
embedding_names = {'CLIPvision','CLIPtext','LLM','MLLM','Human'};

tempFolder = 'data/ROIs';
if ~exist(tempFolder, 'dir')
    mkdir(tempFolder);
end

for sub_id = 1:length(subj_id)
    subj = ['Subject_', num2str(subj_id(sub_id))]
    samples_volume = load(['data/ROIs/', subj, '/fmri_44_all_rois_4d.mat']);
    samples_volume = samples_volume.data;

    samples_volume_trial_1 = load(['data/ROIs/', subj, '/fmri_44_all_rois_4d_trial_1.mat']);
    samples_volume_trial_1 = samples_volume_trial_1.data;

    samples_volume_trial_2 = load(['data/ROIs/', subj, '/fmri_44_all_rois_4d_trial_2.mat']);
    samples_volume_trial_2 = samples_volume_trial_2.data;

    samples_volume_trial_3 = load(['data/ROIs/', subj, '/fmri_44_all_rois_4d_trial_3.mat']);
    samples_volume_trial_3 = samples_volume_trial_3.data;


    sample_data = squeeze(samples_volume(1,:,:,:));
    non_zero_count = nnz(sample_data);

    nz_voxels = find(sample_data ~= 0);
    xyz = size(sample_data);
    subj_scores = zeros(length(embedding_names),xyz(1),xyz(2),xyz(3),repeat);

    data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
    spose_embedding_chatgpt = load(fullfile(data_dir,'spose_embedding_nsd44_predicted_from_chatgpt.txt'));

    data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
    spose_embedding_gemini = load(fullfile(data_dir,'spose_embedding_nsd44_predicted_from_gemini.txt'));

    data_dir = fullfile(base_dir,'data/Humans/');
    spose_embedding_human = load(fullfile(data_dir,'spose_embedding_nsd44_predicted_from_humans.txt'));


    load(fullfile(base_dir,'data/ROIs/44image_feas_CLIP_ViT-L_14.mat'));
    data_CLIPvision = data(re_rankindex,:);
    load(fullfile(base_dir,'data/ROIs/44image_caption_feas_CLIP_ViT-L_14.mat'));
    data_CLIPtext = data(re_rankindex,:);

    for voxel_idx = 1:numel(nz_voxels)
        
        [x, y, z] = ind2sub(size(sample_data), nz_voxels(voxel_idx));

        sphere_voxels = [];
        for i = x-3:x+3
            for j = y-3:y+3
                for k = z-3:z+3
                    if i >= 1 && i <= size(sample_data, 1) && j >= 1 && j <= size(sample_data, 2) && k >= 1 && k <= size(sample_data, 3)
                        sphere_voxels = [sphere_voxels samples_volume(:, i, j, k)];
                    end
                end
            end
        end

        for bs = 1:repeat

            sampled_idx = datasample(label_idx, length(label_idx), 'Replace', true);

            data = sphere_voxels(re_rankindex, :);
            data_sampled = data(sampled_idx, :);
            RSM44_brain = corrcoef(data_sampled');


            % load chatgpt embedding
            % for that focus on those 44 objects only rather than the entire matrix
            spose_embedding = spose_embedding_chatgpt(re_rankindex,:);
            spose_embedding_sampled = spose_embedding(sampled_idx, :);
            dot_product44 = spose_embedding_sampled*spose_embedding_sampled';
            objectposition44 = [1:44];
            esim = exp(dot_product44);
            cp = zeros(44,44);
            ctmp = zeros(1,44);
            for i = 1:44
                for j = i+1:44
                    ctmp = zeros(1,44);
                    for k_ind = 1:length(objectposition44)
                        k = objectposition44(k_ind);
                        if k == i || k == j, continue, end
                        ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
                    end
                    cp(i,j) = sum(ctmp); % run sum first, divide all by 44 later
                end
            end
            cp = cp/44; % complete the mean
            cp = cp+cp'; % symmetric
            cp(logical(eye(size(cp)))) = 1;
            spose_sim44_chatgpt = cp;


            % load gemini embedding
            % for that focus on those 44 objects only rather than the entire matrix
            spose_embedding = spose_embedding_gemini(re_rankindex,:);
            spose_embedding_sampled = spose_embedding(sampled_idx, :);
            dot_product44 = spose_embedding_sampled*spose_embedding_sampled';
            objectposition44 = [1:44];
            esim = exp(dot_product44);
            cp = zeros(44,44);
            ctmp = zeros(1,44);
            for i = 1:44
                for j = i+1:44
                    ctmp = zeros(1,44);
                    for k_ind = 1:length(objectposition44)
                        k = objectposition44(k_ind);
                        if k == i || k == j, continue, end
                        ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
                    end
                    cp(i,j) = sum(ctmp); % run sum first, divide all by 44 later
                end
            end
            cp = cp/44; % complete the mean
            cp = cp+cp'; % symmetric
            cp(logical(eye(size(cp)))) = 1;
            spose_sim44_gemini = cp;


            % load human embedding
            % for that focus on those 44 objects only rather than the entire matrix
            spose_embedding_sampled = spose_embedding_human(sampled_idx, :);
            dot_product44 = spose_embedding_sampled*spose_embedding_sampled';
            objectposition44 = [1:44];
            esim = exp(dot_product44);
            cp = zeros(44,44);
            ctmp = zeros(1,44);
            for i = 1:44
                for j = i+1:44
                    ctmp = zeros(1,44);
                    for k_ind = 1:length(objectposition44)
                        k = objectposition44(k_ind);
                        if k == i || k == j, continue, end
                        ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
                    end
                    cp(i,j) = sum(ctmp); % run sum first, divide all by 44 later
                end
            end
            cp = cp/44; % complete the mean
            cp = cp+cp'; % symmetric
            cp(logical(eye(size(cp)))) = 1;
            spose_sim44_human = cp;


            data_sampled = data_CLIPvision(sampled_idx, :);
            sim44_CLIP_visual = corrcoef(data_sampled');


            data_sampled = data_CLIPtext(sampled_idx, :);
            sim44_CLIP_language = corrcoef(data_sampled');


            spose_sim44={sim44_CLIP_visual,sim44_CLIP_language,spose_sim44_chatgpt,spose_sim44_gemini,spose_sim44_human};

            for k = 1:length(embedding_names)
                subj_scores(k,x,y,z,bs) = corr(squareformq(spose_sim44{k}), squareformq(RSM44_brain));
            end

        end
     
    end
    for model = 1:length(embedding_names)
        model_name = embedding_names(model);
        model_data = squeeze(subj_scores(model,:,:,:,:));
        save_filename = [tempFolder,'/rsa_scores_', subj,'_',model_name{1},'_','bootstrap','.mat'];  
        save(save_filename, 'model_data'); 
    end
end
