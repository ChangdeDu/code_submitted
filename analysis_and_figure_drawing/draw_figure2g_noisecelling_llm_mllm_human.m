% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))


%% ChatGPT-3.5: Predict behavior and similarity
%% Load relevant data
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile(data_dir,'triplet_dataset\validationset_ChatGPT_3.5.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata(triplet_testdata==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata = triplet_testdata-10000;

dosave = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product66(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_llm = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_llm = 1.96*std(behav_predict_obj)/sqrt(1854);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
h = fopen(fullfile(data_dir,'triplets_noiseceiling_ChatGPT_table.csv'),'r');
NCdat = zeros(20000,4);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000)

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
fig = figure('Position',[800 800 600 500],'color','none');
% first plot noise ceiling
x =  [2 6 6 2];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on


%% Gemini_Pro_Vision: Predict behavior and similarity
%% Load relevant data
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile(data_dir,'triplet_dataset\validationset_Gemini_Pro_Vision.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata(triplet_testdata==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata = triplet_testdata-10000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product66(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_vlm = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_vlm = 1.96*std(behav_predict_obj)/sqrt(1854);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
h = fopen(fullfile(data_dir,'triplets_noiseceiling_Gemini_Vision_table.csv'),'r');
NCdat = zeros(20000,4);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000)

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
x = [8 12 12 8];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on

%% Humans: Predict behavior and similarity
%% Load relevant data
data_dir = fullfile(base_dir,'data/Humans');
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile(data_dir,'triplet_dataset\validationset.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata(triplet_testdata==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata = triplet_testdata-10000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product66(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95 = 1.96*std(behav_predict_obj)/sqrt(1854);


%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%

h = fopen(fullfile(data_dir,'triplets_noiseceiling.csv'),'r');
NCdat = zeros(20000,5);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000)

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%

% first plot noise ceiling
x = [14 18 18 14];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on

% now plot results
ha3 = bar(4,behav_predict_acc_llm,'FaceColor',[.2 .6 .4],'EdgeColor','none','BarWidth',4);
hb3 = errorbar(4, behav_predict_acc_llm, behav_predict_acc_ci95_llm,'Color',[0 0 0],'LineWidth',4);

ha4 = bar(10,behav_predict_acc_vlm,'FaceColor',[.6 .2 .4],'EdgeColor','none','BarWidth',4);
hb4 = errorbar(10, behav_predict_acc_vlm, behav_predict_acc_ci95_vlm,'Color',[0 0 0],'LineWidth',4);

ha5 = bar(16, behav_predict_acc,'FaceColor',[.2 .4 .6],'EdgeColor','none','BarWidth',4);
hb5 = errorbar(16, behav_predict_acc, behav_predict_acc_ci95,'Color',[0 0 0],'LineWidth',4);

hb = plot([-1,20],[33.3333 33.3333],'--r','LineWidth',3);

% text(1.6, 63.5, 'LLM noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 13);
% text(7, 69.3, 'VLM noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 13);
text(14, 69.3, ' noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 13);
text(14.0, 35, 'chance', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
text(4, 47, 'ChatGPT-3.5', 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize', 14);
text(10, 47, 'Gemini Pro Vision', 'Rotation',90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize',  14);
text(16, 47, 'Human', 'Rotation',90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize',  14);
axis equal
xlim([-1,20])
ylim([30 76])

% 添加纵向标题
ylabel('Prediction accuracy (%)', 'FontSize', 16);

% 添加横向标题并居中
xlabel('Model performance', 'FontSize', 16);

hax = gca;
set(gca,'FontSize',12) % 字号可以根据需要进行调整
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1;
hax.Box = 'off';

if dosave
    exportgraphics(fig, 'noisecelling_llm_mllm_human.pdf', 'ContentType', 'vector');
end
close(fig);
