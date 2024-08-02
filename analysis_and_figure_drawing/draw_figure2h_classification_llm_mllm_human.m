
% run this script from where it is located
base_dir = pwd;

variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))


%% classification analysis (no figure involved but included for reproducibility)
% results printed to screen 
embedding = 'spose_embedding_66d_sorted_humans.txt';
data_dir = fullfile(base_dir,'data/Humans/');
acc_human = predict_category(embedding, base_dir, data_dir,variable_dir);

embedding = 'spose_embedding_66d_sorted_chatgpt.txt';
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
acc_chatgpt = predict_category(embedding, base_dir, data_dir,variable_dir);

embedding = 'spose_embedding_66d_sorted_gemini.txt';
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
acc_gemini = predict_category(embedding, base_dir, data_dir,variable_dir);


%% Load relevant data


%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
fig = figure('Position',[800 800 600 500],'color','none');

% now plot results
ha3 = bar(11,acc_chatgpt,'FaceColor',[.2 .6 .4],'EdgeColor','none','BarWidth',9);
hold on;
ha4 = bar(25,acc_gemini,'FaceColor',[.6 .2 .4],'EdgeColor','none','BarWidth',9);
ha5 = bar(39, acc_human,'FaceColor',[.2 .4 .6],'EdgeColor','none','BarWidth',9);
hb = plot([-1,46],[5.555 5.555],'--r','LineWidth',3);

text(31.0, 9, 'chance', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 18);
text(11, 47, 'ChatGPT-3.5', 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize', 18);
text(25, 47, 'Gemini Pro Vision', 'Rotation',90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize',  18);
text(39, 47, 'Human', 'Rotation',90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize',  18);
axis equal
xlim([-1,48])
ylim([0 90])
ylabel('Categorization performance (%)', 'FontSize', 18);

hax = gca;
set(gca,'FontSize',14) 
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1;
hax.Box = 'off';

exportgraphics(fig, 'classification_llm_vlm_human.pdf', 'ContentType', 'vector');
close(fig);
