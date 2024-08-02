
% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data');
variable_dir = fullfile(base_dir,'data/variables');

roi_names = {'EarlyVis','RSC', 'OPA', 'EBA', 'FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca'}; 
subj_id = [1,2,3,4,5,6,7,8];
embedding_names = {'LLM','VLM','Human','CLIPvision','CLIPtext'};
embedding_names_label = {'LLM','MLLM','Human','CLIPvision','CLIPtext'};


roi_avg_scores = zeros(length(embedding_names), length(roi_names));
for i = 1:length(subj_id)
    subj = ['Subject_', num2str(subj_id(i))];
    load(fullfile(data_dir,['ROIs/roi_rsa_scores_', subj, '.mat']));
    roi_avg_scores = roi_avg_scores+data;
end
roi_avg_scores = roi_avg_scores/length(subj_id);


fig = figure('Position',[2500 1000 2500 1000],'color','none');
h = bar(roi_avg_scores');
h(1).FaceColor = [.4 .2 .8]; 
h(2).FaceColor = [.8 .5 .3]; 
h(3).FaceColor = [.2 .6 .4]; 
h(4).FaceColor = [.6 .2 .4]; 
h(5).FaceColor = [.2 .4 .6]; 

xticklabels(roi_names); 
ylabel('Noise-normalized similarity score', 'FontSize', 38);
xlabel('Brain ROIs', 'FontSize', 38);
legend(embedding_names_label, 'Location', 'best', 'FontSize', 32);
title('8 subjects averaged',  'FontSize', 38);
xlim([0.5 11.5])
ylim([0 0.5])

hax = gca;
set(gca,'FontSize',30) 
hax.TickDir = 'both';
% hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1;
hax.Box = 'off';

exportgraphics(fig, 'searchlight_rsa_barplot.pdf', 'ContentType', 'vector');
close(fig);

