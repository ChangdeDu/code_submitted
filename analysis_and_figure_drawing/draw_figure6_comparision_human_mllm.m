% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');

data_dir = fullfile(base_dir,'data/Humans/');
load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));
spose_embedding_human = spose_embedding_66d_sorted_humans;

data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
spose_embedding_gemini = spose_embedding_66d_sorted_gemini;

load(fullfile(variable_dir,'labels_short.mat'))
labels_human = labels_short;
labels_gemini = importdata(fullfile(variable_dir,'labels_short_66_gemini.txt'));

clear labels
c = corr(spose_embedding_human,spose_embedding_gemini);
% now let's sort dimensions by maximal cross-correlation
[~,ii] = max(c);
% remove ones that have already appeared
for i = 2:66
    if any(ii(1:i-1)==ii(i))
        ii(i) = 100;
    end
end
[~,si] = sort(ii);

% now remove original cross-correlation matrix to find differences
c_base = [(corr(spose_embedding_human) - eye(66))];
c_adapted = c(:,si)-c_base;
zeroind = [c_adapted(1:66,1:66)]<0.4;
c_adapted(zeroind) = 0;

for row = 1:size(c_adapted, 1)
    [sortedValues, sortedIndices] = sort(c_adapted(row,:), 'descend');
    c_adapted(row, sortedIndices(3:end)) = 0; 
end


% dimension reproducibility: sorting dimensions by maximal correlation
disp(sort(diag(c_adapted),'descend'))
% only three dimensions not reproduced: handicraft-related and cylindrical
% (sparsest) as well as partial reproduction of dim 32 (flat/patterned),
% which was among the least interpretable dims but which originally was
% quite reproducible

figure('Position',[300 1 800 700])
imagesc(c(:,si))
ax = gca;

ax.YTick = 1:66;
ax.YTickLabels = labels_human;
set(ax, 'YTickLabel', ax.YTickLabels, 'FontSize', 7);
ax.XTick = 1:66;
ax.XTickLabels = labels_gemini(si);
ax.XTickLabelRotation = 90;
set(ax, 'XTickLabel', ax.XTickLabels, 'FontSize', 7);
% colormap('hsv');
colorbar;
set(gcf,'Units','centimeters')
screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)])
set(gcf,'PaperSize',screenposition(3:4))

ylabel('Human dimensions', 'FontSize', 12);
xlabel('MLLM dimensions', 'FontSize', 12);
saveas(gcf, 'reordered_corr_matrix_mllm_human.pdf');

clear x
[x(:,1) x(:,2)] = ind2sub([66 66],find(c_adapted));
% x(:,2) = si(x(:,2));
x = sortrows(x,[1 2]);

%% now extend x to get the size of each one right, and make two more variables to complete the first where variables 
%% without a pair from the other side will be plotted as flat until the middle
n_rep49 = max(histc(x(:,1),1:66));
n_rep66 = max(histc(x(:,2),1:66));
n_rep_total = n_rep49*n_rep66; % the product of both is the safest


%% plot flow diagram

addpath('helper_functions\external\patchline')

% start out with a version where we map 49 points to 66 points

pos_x61 = -8;
pos_x66 = 8;
pos_y49 = linspace(1,100,66);
pos_y66 = linspace(1,100,66);

% flow components as sigmoid function from one point to another with fixed
% slope

% given a known starting point and end point, what is this function?
% a is starting point and end point is a+b, so to add start and end point
% in function this should do the job
sigmoid = @(x,a,b) a+ (b-a)./(1+exp(-x));

k = linspace(pos_x61,pos_x66,100);
load(fullfile(variable_dir,'colors66.mat'))
figure('Position',[300 1 800 1100])
hold on
rng(42) % was 4
for i = 1:size(x,1)
    if x(i,1)==x(i,2)
        hl(i) = patchline([k(1) k(end)],[x(i,1) x(i,2)],'edgecolor',colors66(x(i,1),:),'linewidth',8*c_adapted(x(i,1),x(i,2)),'edgealpha',0.3);
    else
    ktmp = linspace(pos_x61+4*randn,pos_x66+4*randn,100);
    ytmp = sigmoid(ktmp,(x(i,1)),(x(i,2)));
    while abs(ytmp(1)-x(i,1))>0.1 || abs(ytmp(end)-x(i,2))>0.1
        ktmp = linspace(pos_x61+2*randn,pos_x66+2*randn,100);
        ytmp = sigmoid(ktmp*randn,(x(i,1)),(x(i,2)));
    end
    hl(i) = patchline(k,ytmp,'edgecolor',colors66(x(i,1),:),'linewidth',8*c_adapted(x(i,1),x(i,2)),'edgealpha',0.7);
    end
end
yl = ylim;
text(9, -1.6, 'MLLM embedding (66 dimensions)', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
text(-9, -1.6, 'Human embedding (66 dimensions)', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
a = gca;
a.XTick = [];
drawnow
a.XRuler.Axle.Visible = 'off'; % a is axis handle
a.YRuler.Axle.Visible = 'off';


set(gca,'YDir','reverse')
set(gca,'YTick',1:66,'YTickLabel',labels_human, 'FontSize', 11)
% set(gca,'ticklength',[0 0])
yyaxis left
ylim(yl)

yyaxis right
ylim(yl)
set(gca,'YDir','reverse')
set(gca,'YTick',1:66,'YTickLabel',labels_gemini(si),'YColor',[0 0 0], 'FontSize', 11)

doprint = 1;
if doprint
print(gcf,'-dpdf','figure_river_mllm_human.pdf','-bestfit') 
end