%% Plots for Paper

clear

% Add paths for toolboxes
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/TOSSH-master/'))
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/marrmot_211/'))
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/ma_thesis/'))
%addpath(genpath(''))

% load data from Results and Benchmark
load('additional.mat')
load('signatures.mat')
load('OF_values.mat')

%objective_functions = obj_funs;
%current_signatures = signatures;

% Define catchments and signatures
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155','camels_12381400','camels_02017500',...
    'camelsgb_39037','camels_03460000','hysets_01AF007','camelsgb_27035','lamah_200048'};
catchments_labels = {'AUS1','BR6','AUS6','C12','C02','GB3','C02','HYS','GB2','LAM'};
signatures_Q = {'sig_FDC_slope','sig_RisingLimbDensity',...
    'sig_BaseflowRecessionK','sig_HFD_mean',...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex'};
signatures_Q_P = {'sig_EventRR','sig_TotalRR'};
signatures_high_low = {'sig_x_Q_duration','sig_x_Q_frequency'};
signatures_percentage = {'sig_x_percentile'};

current_signatures = [signatures_Q,signatures_Q_P,signatures_high_low,signatures_percentage];

% Locate Netcdfs for catchments
path_nc = '/Users/peterwagener/Desktop/ma_thesis_dump/ma_thesis/catchments_new/';
Files=dir(fullfile(path_nc,'*.nc'));
% catchments = cell(length(Files)+1,1);
% for k = 1:length(Files)
%     FileNames=Files(k).name;
%     catchments{k} = FileNames(1:end-3);
% end

% select catchments

% Organize Signatures and Models
sorted_signatures = {'sig_TotalRR','sig_EventRR','sig_x_percentile_5per','sig_x_percentile_95per','sig_x_Q_duration_high','sig_x_Q_duration_low',...
    'sig_x_Q_frequency_high','sig_x_Q_frequency_low','sig_HFD_mean','sig_FDC_slope','sig_VariabilityIndex','sig_BFI','sig_BaseflowRecessionK',...
    'sig_FlashinessIndex','sig_RisingLimbDensity'};

models_detail = {'m_03_collie2_4p_1s','m_07_gr4j_4p_2s','m_31_mopex3_8p_5s','m_34_flexis_12p_5s','m_37_hbv_15p_5s'};
model_name = {'Collie 2','GR4J','Mopex 3','Flexis','HBV'};

% plot benchmark values in boxplot/violinplot
signatures = fieldnames(obs_signatures_cali_bench);

% Make Tags for Plotting 
signature_list_plot = {'FDC Slope','Rising Limb Density (1/d)','Baseflow Recession Coefficient (1/d)','Mean Half Flow Date',...
'Baseflow Index','Variability Index','Flashiness Index','Event RR',...
'Total RR','High Flow Duration (d)','Low Flow Duration (d)','High Flow Frequency',...
'Low Flow Frequency','Q5 (mm/d)','Q95 (mm/d)'};

OF_Plot = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};

OF_Plot_double = [OF_Plot,OF_Plot];

threshold(:,10) = [];

%% Read Data
% read data
% catchments from Caravan (NetCDF)
% ncdisp('/Users/peterwagener/Desktop/ma_thesis/lamah_212670.nc')
%lon = ncread('/Users/peterwagener/Desktop/ma_thesis/lamah_212670.nc','longitude')

% Get selected Caravan Data
precip = zeros(14609,length(catchments_aridity));
temp = zeros(14609,length(catchments_aridity));
pet = zeros(14609,length(catchments_aridity));
streamflow = zeros(14609,length(catchments_aridity));

% write struct/cell with data (precip, temp, pet, streamflow)
for h = 1:10
    precip(:,h)   = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'total_precipitation_sum');
    temp(:,h)     = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'temperature_2m_mean');
    pet(:,h)      = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'potential_evaporation_sum');
    streamflow(:,h) = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'streamflow');
end

%% Select catchments based on benchmark
catchments_labels = {'AUS1','BR6','AUS6','C12','C02','GB3','C03','HYS','GB2','LAM'};
OF_Plot = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};

%colors =(brewermap(8,"Set1"));
colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

model_counter = zeros(8,10);

for catchmentIdx = 1:10
    catchment = catchments_aridity{catchmentIdx};

    for objectiveIdx = 1:8
        obj_fun = objective_functions{objectiveIdx};
        for modelIdx = 1:47
            model = model_list{modelIdx};
            try
                if OF_value_cali.(catchment).(obj_fun).(model) > threshold(objectiveIdx,catchmentIdx)
                    OF_value_vali_benchmark.(catchment).(obj_fun).(model) = OF_value_vali.(catchment).(obj_fun).(model);
                    OF_value_cali_benchmark.(catchment).(obj_fun).(model) = OF_value_cali.(catchment).(obj_fun).(model);
                    model_counter(objectiveIdx,catchmentIdx) = model_counter(objectiveIdx,catchmentIdx) +1;
                else
                    
                end
            catch 

            end
        end
    end
end

model_counter_plot = sum(model_counter);
f=figure('units','normalized','outerposition',[0 0 0.5 0.5]);

%hold on

bh = bar(model_counter(:,:)');%;,'stacked');
set(bh, 'FaceColor', 'Flat')
for k = 1:8
    bh(k).CData = colors(k,:);
end

yline(47,'-','All Models Pass Benchmark')
%yline(0,'-','No Models Pass Benchmark')

fontsize(f,16,"points")

legend(strrep(OF_Plot,'_','\_'),'Location','eastoutside')
xticklabels(strrep(catchments_labels,'_','\_'))
xlabel('Catchments')
ylabel('Number of Models Outperforming Benchmark')
xtickangle(45)
%% Heatmap Model Counts

figure('units','normalized','outerposition',[0 0 0.5 0.5])
hm = heatmap(model_counter');
title('Number of Models Outperforming Benchmark')
xlabel('Objective Functions')
%xticks()

hm.YDisplayLabels = catchments_labels;
hm.XDisplayLabels = OF_Plot;
ylabel('Catchments')
hm.FontSize = 25;
%yticks()
%yticklabels(catchments_labels)
% Number of models for each catchment and OF

%% BOXPLOT of OBJECTIVE FUNCTIONS VALUES FOR EACH catchment and subplot for each OF
f = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
colors = (brewermap(10, "Spectral"));
%colors =(brewermap(8,"Set1"));
colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];


% Calculate the number of rows and columns based on subplots (assuming 4x2 layout)
num_rows = 2;
num_cols = 4;

% Iterate over objective functions and create subplots
for j = 1:numel(objective_functions)
    obj_fun = objective_functions{j};
    X = [];
    G = [];
    
    % Create a subplot for the current objective function
    ax = subplot(num_rows, num_cols, j);
    
    % Initialize a cell array to store the boxplot data for each catchment
    boxplot_data = cell(1, numel(catchments_aridity));
    
    % Iterate over catchments and collect the model values for each catchment
    for i = 1:numel(catchments_aridity)
        catchment = catchments_aridity{i};
        
        % Initialize an array to store the model values for the current catchment
        catchment_values = [];
        
        % Iterate over models and collect the values
        for k = 1:numel(model_list)
            model = model_list{k};
            
            % Get the value of the objective function for the current model and catchment
            try
                value = OF_value_cali.(catchment).(obj_fun).(model);
                catchment_values = [catchment_values; value];
            catch
            end
        end
        
        % Store the catchment_values array in the boxplot_data cell array
        boxplot_data{i} = catchment_values;
        if ~isempty(catchment_values)
            X = [X; catchment_values];
            G = vertcat(G, (i) * ones(size(catchment_values)));
        else
            X = [X; threshold(j, i) - 3]; % Add a NaN value to X for each empty catchment
            G = [G; (i)]; % Add a corresponding grouping value to G
        end
    end
    
    % Plot the threshold values
    for i = 1:length(catchments_aridity)
        line('XData', i, 'YData', threshold(j,i), 'Marker', '_', 'MarkerEdgeColor', 'r', 'MarkerSize', 20, 'LineWidth', 3);
    end
    hold on
    
    % Create the violin plots
    vp = violinplot(X, G, 'ViolinColor', colors(j,:));
    for v = 1:length(vp)
        vp(v).ScatterPlot.SizeData = 10;   % default is usually ~36
    end
    
    set(ax, 'XTick', 1:numel(catchments_aridity), 'XTickLabel', catchments_labels); % Keep catchment names
    
    % Conditionally remove only the x-axis label (not the tick labels) for non-bottom row subplots
    if j <= num_cols * (num_rows - 1)
        set(get(gca, 'xlabel'), 'String', ''); % Remove the x-axis label
    else
        xtickangle(45); % Keep the x-tick angle for the bottom row
        xlabel('Catchments')
    end
    
    % Conditionally remove only the y-axis label (not the tick labels) for non-left column subplots
    if mod(j, num_cols) ~= 1
        set(get(gca, 'ylabel'), 'String', ''); % Remove the y-axis label
    else
        ylabel('Objective Function Value'); % Keep y-label for the first column
    end
    
    % Adjust the title position
    title(['Objective Function: ' OF_Plot{j}], 'Interpreter', 'none', ...
          'Position', [5, 1.1, 0]); % Move the title up (adjust Y-coordinate as needed)
    
    ylim([-0.5 1])
    xlim([0.5 10.5])
    grid on;
    set(ax, 'PositionConstraint', 'innerposition')
    
    % Add text annotations above the violin plots
    for i = 1:numel(catchments_aridity)
        text(i, 1.05, num2str(model_counter(j, i)), 'HorizontalAlignment', 'center', 'FontSize', 5, 'Color', 'k');
    end
end

% Adjust the figure size
fig = gcf;
fontsize(f, 10, "points");
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/benchmark_violinplot_new.jpg');
exportgraphics(f, filename, 'Resolution', 300)

%% show observed signature values a 4x4 subplot
sig_values_avg = NaN(length(signatures),length(catchments_aridity));

for catchmentIdx = 1:10
    catchment = catchments_aridity{catchmentIdx};
    for signatureIdx = 1:length(sorted_signatures)
        signature = sorted_signatures{signatureIdx};
        try
            % Calculate the percentual error for the current catchment, model, and signature
            sig_values_avg(signatureIdx,catchmentIdx) = obs_signatures_eval_bench.(signature).(catchment);
                    
        catch
        end
    end
end
    
%aridity_values = [-0.39,-0.28,-0.04,0.02,0.03,0.16,0.33,0.34,0.42,0.52,0.58];
%seasonality_values = [0.59,1.30,1.67,1.66,1.07,1.46,0.85,1.28,1.25,1.15,0.58];
%snow_values = [0,0,0,0.49,0.07,0,0.08,0.38,0,0.30,0.32];

aridity_values = [-0.39,-0.28,-0.04,0.02,0.03,0.16,0.33,0.34,0.42,0.58];
seasonality_values = [0.59,1.30,1.67,1.66,1.07,1.46,0.85,1.28,1.25,0.58];
snow_values = [0,0,0,0.49,0.07,0,0.08,0.38,0,0.32];


signature_list_plot_sorted = {'Total RR','Event RR','Q5 (mm/d)','Q95 (mm/d)',...
'High Flow Duration (d)','Low Flow Duration (d)','High Flow Frequency',...
'Low Flow Frequency','Mean Half Flow Date (DOY)','FDC Slope','Variability Index','Baseflow Index',...
'BF Recession Coefficient (1/d)','Flashiness Index','Rising Limb Density (1/d)'};

%range_min = [NaN,NaN,0,     0,      NaN,0,0,0,0,NaN,NaN,0,0,NaN,NaN];
%range_max = [NaN,NaN,0.5,   365,    NaN,1,1,1,1,NaN,NaN,1,1,NaN,NaN];

range_min = [0,0,NaN,NaN,NaN,NaN,0,0,0,     NaN,0,0,NaN,0,0];
range_max = [1,1,NaN,NaN,NaN,NaN,1,1,365,   NaN,1,1,NaN,1,1];
catchments_seasonality = {};

% Check signaficance

% Number of observations (rows)
n = size(sig_values_avg, 1);

% Number of estimated parameters (in this case, only the slope)
num_parameters = 1;

% Significance level (alpha)
alpha = 0.05;

% Preallocate arrays to store results
t_values = zeros(1, n);
p_values = zeros(1, n);

for i = 1:n
    y = sig_values_avg(i, :); % Dependent variable (response)
    x = 1:length(y); % Independent variable (predictor)
    
    % Handle NaN values
    valid_indices = ~isnan(y);
    y_valid = y(valid_indices);
    x_valid = x(valid_indices);

    % Perform linear regression
    mdl = fitlm(x_valid, y_valid);
    
    % Get the coefficient for the slope (regression coefficient)
    slope_coefficient = mdl.Coefficients.Estimate(2);
    
    % Calculate the standard error of the coefficients
    se_coefficient = mdl.Coefficients.SE(2);
    
    % Calculate the t-value
    t_value = slope_coefficient / se_coefficient;
    
    % Calculate the p-value
    p_value = 2 * (1 - tcdf(abs(t_value), n - num_parameters));
    
    % Store results
    t_values(i) = t_value;
    p_values(i) = p_value;
end

% Adjust p-values for multiple testing (if needed)
adjusted_p_values = p_values * n;

% Find significant trends based on adjusted p-values
significant_trends = find(p_values < alpha);

disp("Row numbers with significant trends:");
disp(significant_trends);
% Plot
figure('units','normalized','outerposition',[0 0 1 1]) 

for i = 1:15
    ax = subplot(4,4,i);
    
    scatter(aridity_values,sig_values_avg(i,:),'x')
    xlabel('Aridity')
    xlim([-1 1])
    ylabel('Signature Value')

    % Remove NaN values from the data
    valid_data_indices = ~isnan(aridity_values) & ~isnan(sig_values_avg(i, :));
    aridity_vec = aridity_values(valid_data_indices);
    sig_avg_vec = sig_values_avg(i, valid_data_indices)';
    
    % Calculate the regression line coefficients
    coeffs = polyfit(aridity_vec, sig_avg_vec, 1);
    x_fit = [min(aridity_values) max(aridity_values)];
    y_fit = polyval(coeffs, x_fit);
    
    % Plot the regression line
    hold on;
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
    grid on
    
    if isnan(range_min(i))
        
    else
        ylim([range_min(i) range_max(i)])
    end

    hold off;

    P_str = sprintf('p=%.2f',p_values(i));
    annotation('textbox',[0 0 0.3 0.3],'String',P_str,'Position',ax.Position,'Vert','top','FitBoxToText','on')
    %TextLocation(P_str,'Location','best')

    % Add title to the subplot
    title(strrep(signature_list_plot_sorted{i}, '_', '\_'));

end
f = gcf;

fontsize(f,16,"points")
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/signature_obs.jpg');
exportgraphics(f,filename,'Resolution',300)
%% climate plot
figure();

subplot(3,1,1);
plot(aridity_values,aridity_values,'o-');
ylim([-1 1])
xlim([-1 1])
grid on
ylabel('Aridity')

subplot(3,1,2)
plot(aridity_values,seasonality_values,'o-');
ylim([0 2])
xlim([-1 1])
grid on 
ylabel('Seasonality')

ax = subplot(3,1,3);
plot(aridity_values,snow_values,'o-');
ylim([0 1]) 
xlim([-1 1])
grid on 
ylabel('Snow Tendency')
xlabel('Aridity')
%xticks();
%set(ax, 'XTick', 1:numel(catchments_aridity), 'XTickLabel', strrep(catchments_aridity,'_', '\_'));
%xtickangle(45);    

%% Calculate signatures

% Specific signature % error for each catchment

model_list_sorted = {'m_01_collie1_1p_1s', 'm_02_wetland_4p_1s', 'm_03_collie2_4p_1s',...
    'm_06_alpine1_4p_2s','m_07_gr4j_4p_2s','m_17_penman_4p_3s',  'm_08_us1_5p_2s','m_24_mopex1_5p_4s','m_29_hymod_5p_5s',...
    'm_04_newzealand1_6p_1s','m_09_susannah1_6p_2s', 'm_10_susannah2_6p_2s','m_11_collie3_6p_2s',  'm_12_alpine2_6p_2s',...
    'm_25_tcm_6p_4s', 'm_05_ihacres_7p_1s', 'm_13_hillslope_7p_2s', 'm_14_topmodel_7p_2s', 'm_18_simhyd_7p_3s', 'm_30_mopex2_7p_5s',...
    'm_15_plateau_8p_2s',  'm_16_newzealand2_8p_2s','m_19_australia_8p_3s', 'm_20_gsfb_8p_3s','m_31_mopex3_8p_5s', 'm_40_smar_8p_6s',...
    'm_21_flexb_9p_3s',...
    'm_22_vic_10p_3s','m_26_flexi_10p_4s', 'm_32_mopex4_10p_5s',  'm_41_nam_10p_6s',...
    'm_33_sacramento_11p_5s',...
    'm_27_tank_12p_4s', 'm_28_xinanjiang_12p_4s', 'm_34_flexis_12p_5s',  'm_35_mopex5_12p_5s',  'm_42_hycymodel_12p_6s','m_43_gsmsocont_12p_6s','m_46_classic_12p_8s',...
    'm_36_modhydrolog_15p_5s', 'm_37_hbv_15p_5s',...
    'm_47_IHM19_16p_4s','m_38_tank2_16p_5s','m_39_mcrm_16p_5s',  'm_44_echo_16p_6s',...
    'm_45_prms_18p_7s', ...
    'm_23_lascam_24p_3s'};

data = zeros(length(model_list),length(catchments),length(objective_functions));

for p = 10%length(signatures)
    signature = signatures{p};
    
    for k = 1:length(catchments_aridity) % number of catchments (currently: 3)
        catchment = catchments_aridity{k};
        %figure(Name="Percentage variation of sim from obs for "+signature+' and '+catchment);
        for i = 1:length(model_list) % number of models (currently: 3)
            for j = 1:length(objective_functions)
                
                
                model = model_list_sorted{i};
                obj_fun = objective_functions{j};
                
    
                try
                    data(i,k,j) = (sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment))/...
                    obs_signatures_cali_bench.(signature).(catchment)*100;
                catch
                   fprintf('Data not found: %s\n', [signature, '.', catchment, '.', model, '.', obj_fun]);
                   data(i,k,j) = NaN;
                end
            end
        end
    end
end
%% Extra plot defense
figure('units','normalized','outerposition',[0 0 1 1]);
C = colororder;
C(8,:) = [0.1, 0.1, 0.1];

sorted_signatures_short = {'sig_TotalRR'	,'sig_EventRR','sig_x_percentile_5per','sig_x_percentile_95per',...
    'sig_HFD_mean',	'sig_FDC_slope'	,	'sig_BFI','sig_FlashinessIndex',	'sig_RisingLimbDensity'};

label_signatures = {'Total RR','Event RR','Q5','Q95','MHFD','FDC Slope','BFI','Flashiness Index','Rising Limb Density'};

store_defense = NaN(4,9);
for i = 1:9
    signature = sorted_signatures_short{i};
    subplot(3,3,i)
    for j = 1:8
        obj_fun = objective_functions{j};
        
        store_temp = NaN(11,2);
        for k = 1:10
            catchment = catchments_aridity{k};
            store_temp_model = NaN(47,1);
            for l = 1:47
                model = model_list{l};
                try
                    if obs_signatures_cali_bench.(signature).(catchment)~=0
                        store_temp_model(l,1) = 100*(sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment))/obs_signatures_cali_bench.(signature).(catchment);
                    else
                        store_temp_model(l,1) = NaN;
                    end
                %store_temp_model(l,2) = obs_signatures_cali_bench.(signature).(catchment);
                catch
                end
            end
            store_temp(k,1) = nanmedian(store_temp_model(:,1));
            store_temp(k,2) = obs_signatures_cali_bench.(signature).(catchment);
        end
        store_defense(i,j)= nanmean(store_temp(:,1));


    end
        for loop = 1:length(objective_functions)
            scatter(loop,store_defense(i,loop),100,C(loop,:),'filled')
            hold on
        end
        %scatter(2,store_defense(i,2),100,C(2,:),'filled')
        %scatter(3,store_defense(i,3),100,C(3,:),'filled')
        %scatter(4,store_defense(i,4),100,C(4,:),'filled')
        ylim([-100 100])

        xticks(1:1:8);
        xticklabels(OF_Plot);

        switch i
            case{1,4,7}
                ylabel('Relative Error (%)')
        end
        xlim([0.5 8.5])
        title(label_signatures{i})
        grid on
        fontsize(gcf,20,"points")
        sgtitle('Mean Relative Error of Signature Representation for Median Model Performance')
end
f = gcf;
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/overview_defense.jpg');
exportgraphics(f,filename,'Resolution',300)

%% Numeric Error 
Impact_Signatures = NaN(length(sorted_signatures),4);
Error_Values = NaN(15,8,47,10);
Norm_Values = NaN(15,1);
Error_Norm_Values = NaN(15,10,8);
Error_Norm_Values_range = NaN(15,10,8);
Signature_cal_eval = NaN(15,10,8);
Signature_cal_median = NaN(15,10,8);
median_catchment_error = NaN(15,8);
median_catchment_error_range = NaN(16,8);
mean_catchment_error_range = NaN(16,8);
norm_plot = NaN(15,10,2);
Rank_values = NaN(15+1,10+1,8);
Error_Values = NaN(15,8,47,10);
Signatures_Array = NaN(15,8,47,10);
Signature_norm = NaN(15,8,47,10);
Signatures_Array_comp = NaN(15,8,47,10);
Signatures_Array_vali = NaN(15,8,47,10);
Signature_eval_median = NaN(9,10,8);

sorted_sig_new = {'sig_TotalRR','sig_x_percentile_5per','sig_x_Q_duration_low','sig_x_Q_frequency_low',...
    'sig_x_percentile_95per','sig_x_Q_duration_high','sig_x_Q_frequency_high','sig_HFD_mean','sig_FDC_slope'...
    'sig_EventRR','sig_BFI','sig_BaseflowRecessionK',...
    'sig_FlashinessIndex','sig_VariabilityIndex','sig_RisingLimbDensity'};

label_signatures_new = {'Total RR','Q5','LF Dur','LF Freq'...
    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
    'Flashiness Index','Variability Index','Rising Limb Density'};

sorted_sig_new = { ...
    'sig_TotalRR', 'sig_EventRR',  'sig_HFD_mean', 'sig_FDC_slope',...  % 1st row
    'sig_x_percentile_5per', 'sig_x_Q_duration_low', 'sig_x_Q_frequency_low', 'sig_BFI', ... % 2nd row
    'sig_x_percentile_95per', 'sig_x_Q_duration_high', 'sig_x_Q_frequency_high', 'sig_BaseflowRecessionK', ... % 3rd row
    'sig_RisingLimbDensity', 'sig_FlashinessIndex', 'sig_VariabilityIndex'}; % 4th row

% Define Y-axis range limits for each signature
yrange_min = [0, 0, -25, 80, ...    % 1st row
              0, 0, 0, 0, ...     % 2nd row
              0, 0, 0, 0, ...   % 3rd row
              0, 0, 0];           % 4th row
              
yrange_max = [1, 0.7, 0, 320, ...   % 1st row
              1.2, 80, 1, 1, ...     % 2nd row
              10, 40, 0.6, 0.4, ...   % 3rd row
              1.5, 1, 1.5];          % 4th row


sorted_sig_new = { ...
    'sig_TotalRR', 'sig_EventRR', 'sig_HFD_mean', ...  % (1st row)
    'sig_FDC_slope', 'sig_x_percentile_5per', 'sig_BaseflowRecessionK', ... % (2nd row)
    'sig_x_Q_duration_high', 'sig_x_Q_frequency_high', 'sig_FlashinessIndex', ... % (3rd row)
    'sig_x_Q_duration_low', 'sig_x_Q_frequency_low', 'sig_BFI', ... % (4th row)
    'sig_x_percentile_95per', 'sig_RisingLimbDensity', 'sig_VariabilityIndex'}; % (5th row)

label_signatures_new = { ...
    'Total RR (-)', 'Event RR (-)', 'FDC Slope (-)', 'MHFD (DOY)', ...             % 1st row
    'Q5 (mm/d)', 'LF Dur (days)', 'LF Freq (-)', 'BFI (-)', ...               % 2nd row
    'Q95 (mm/d)', 'HF Dur (days)', 'HF Freq (-)', 'BFRC (-)', ...             % 3rd row
    'Rising Limb Density (-)', 'Flashiness Index (-)', 'Variability Index (-)'}; % 4th row



% Define labels for signatures
sorted_sig_new = { ...
    'sig_TotalRR', 'sig_EventRR',  'sig_HFD_mean', ...  % 1st row
    'sig_x_percentile_95per', 'sig_x_Q_frequency_high','sig_x_Q_duration_high',   ... % 3rd row
    'sig_x_percentile_5per', 'sig_x_Q_frequency_low','sig_x_Q_duration_low',   ... % 2nd row
    'sig_BFI','sig_BaseflowRecessionK','sig_FDC_slope',... % 4th row
     'sig_FlashinessIndex', 'sig_VariabilityIndex', 'sig_RisingLimbDensity'}; % 5th row

label_signatures_new = { ...
    'Total RR (-)', 'Event RR (-)',  'MHFD (DOY)', ...             % 1st row
    'Q95 (mm/d)',  'HF Freq (-)','HF Dur (days)',  ...             % 3rd row
    'Q5 (mm/d)', 'LF Freq (-)','LF Dur (days)',   ...               % 2nd row
    'BFI (-)', 'BFRC (-)','FDC Slope (-)',... % 4th row
    'Flashiness Index (-)', 'Variability Index (-)', 'Rising Limb Density (-)'}; % 5th row



sorted_sig_new2 = {'sig_TotalRR','sig_x_percentile_5per',...
    'sig_x_percentile_95per','sig_HFD_mean','sig_FDC_slope'...
    'sig_EventRR','sig_BFI',...
    'sig_FlashinessIndex','sig_RisingLimbDensity'};

label_signatures_new2 = {'Total RR','Q5',...
    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
    'Flashiness Index','Rising Limb Density'};

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    norm_values_store = NaN(11,1);
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        for k = 1:length(objective_functions) % number of models (currently: 3)
            %model = model_list_sorted{i};
            obj_fun = objective_functions{k};
            for l = 1:numel(model_list)
                model = model_list{l};
                try 
                    Error_Values(i,k,l,j) = sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment);
                    Signatures_Array(i,k,l,j) = sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun);
                    Signatures_Array_vali(i,k,l,j) = sim_signatures_eval_bench.(signature).(catchment).(model).(obj_fun);%-obs_signatures_cali_bench.(signature).(catchment);
                    Signatures_Array_comp(i,k,l,j) = (Signatures_Array(i,k,l,j)- Signatures_Array_vali(i,k,l,j))./Signatures_Array(i,k,l,j);

                    %if obs_signatures_cali_bench.(signature).(catchment) ~= 0
                            
                        %/obs_signatures_cali_bench.(signature).(catchment);
                    %else
                    %end
                catch
                end
            end
        end
        norm_values_store(j) = obs_signatures_cali_bench.(signature).(catchment);
    end
    Norm_Values(i) = max(norm_values_store)-min(norm_values_store);
    for j=1:10
        for k=1:8
            
            %Error_Norm_Values(i,j,k) = median(Error_Values(i,k,:,j),3,'omitnan')./Norm_Values(i);
            % New normalization using the maximum error
            Error_Norm_Values(i,j,k) = median(Error_Values(i,k,:,j),3,'omitnan');
            Signature_cal_median(i,j,k) = median(Signatures_Array(i,k,:,j),3,'omitnan');
            Signature_cal_eval(i,j,k) = median((Signatures_Array(i,k,:,j)-Signatures_Array_vali(i,k,:,j)),3,'omitnan');
            %median_catchment_error(i,k) = median(Error_Norm_Values(i,:,k),"all",'omitnan');
        end

        %norm_plot(i,j,1) = min(Error_Norm_Values(i,j,:),[],[3],'omitnan');
        norm_plot(i,j,2) = max(abs(Error_Norm_Values(i,j,:)),[],[3],'omitnan');
    end


    for j=1:10
        for k=1:8
            %Error_Norm_Values_range(i,j,k) = (Error_Norm_Values(i,j,k) - norm_plot(i,1))/(norm_plot(i,2)-norm_plot(i,1));
            Error_Norm_Values_range(i,j,k) = (Error_Norm_Values(i,j,k))/(norm_plot(i,j,2));
            Signature_norm(i,k,:,j) = Error_Values(i,k,:,j)/(norm_plot(i,j,2));
            median_catchment_error_range(i+1,k) = median(abs(Error_Norm_Values_range(i,:,k)),"all",'omitnan');
            mean_catchment_error_range(i+1,k) = mean(abs(Error_Norm_Values_range(i,:,k)),"all",'omitnan');

        end
        % Rank the Error_Norm_Values for each signature across all objective functions
        
        %[~, nonnanrank] = sort(A(validmask));
        %ranks = NaN(size(A));
        %rank(validmask) = nonnanrank;

        validmask = ~isnan(Error_Norm_Values(i,j,:));
        sort_array = Error_Norm_Values(i,j,:);
        [~, ranked_indices] = sort(sort_array(validmask)); % Sort to get the ranking
        ranks_1 = NaN(size(Error_Norm_Values(i,j,:)));
        ranks_1(validmask) = ranked_indices;
        
        validmask_2 = ~isnan(ranks_1);
        [~, ranks] = sort(ranks_1(validmask_2)); % Get the ranks (1 for lowest value)
        ranks_2 = NaN(size(Error_Norm_Values(i,j,:)));
        ranks_2(validmask_2) = ranks;

        Rank_values(i,j,1:length(ranks_2)) = ranks_2; % Store the ranks

        for k = 1:8

            Rank_values(16,j,k) = mean(Rank_values(1:15,j,k),'omitnan');
            Rank_values(i,12,k) = mean(Rank_values(i,1:11,k),'omitnan');
            Rank_values(16,12,k) = NaN;

            
            %Error_Norm_Values_range(10,j,k) = std(Error_Norm_Values_range(1:9,j,k),'omitnan');
            %Error_Norm_Values_range(i,12,k) = std(Error_Norm_Values_range(i,1:11,k),'omitnan');
            %Error_Norm_Values_range(10,12,k) = NaN;
        end
    end

end

%% Comparison Plot with underlying violin

% -------------------------------------------------------------------------
% 1) Adjust the new signature order (5 rows x 3 columns)
% -------------------------------------------------------------------------
% sorted_sig_new = { ...
%     'sig_TotalRR', 'sig_EventRR', 'sig_HFD_mean', ...  % (1st row)
%     'sig_FDC_slope', 'sig_x_percentile_5per', 'sig_BaseflowRecessionK', ... % (2nd row)
%     'sig_x_Q_duration_high', 'sig_x_Q_frequency_high', 'sig_FlashinessIndex', ... % (3rd row)
%     'sig_x_Q_duration_low', 'sig_x_Q_frequency_low', 'sig_BFI', ... % (4th row)
%     'sig_x_percentile_95per', 'sig_RisingLimbDensity', 'sig_VariabilityIndex'}; % (5th row)
% 
% % -------------------------------------------------------------------------
% % 2) Define user-friendly labels in the same new order
% % -------------------------------------------------------------------------
% %label_signatures_new = { ...
%     'Total RR (-)', ...
%     'Event RR (-)', ...
%     'MHFD (DOY)', ...
%     'FDC Slope (-)', ...
%     'Q5 (mm/d)', ...
%     'BFRC (-)', ...
%     'HF Dur (days)', ...
%     'HF Freq (-)', ...
%     'Flashiness Index (-)', ...
%     'LF Dur (days)', ...
%     'LF Freq (-)', ...
%     'BFI (-)', ...
%     'Q95 (mm/d)', ...
%     'Rising Limb Density (-)', ...
%     'Variability Index (-)'};

% -------------------------------------------------------------------------
% 3) Define Y-axis range limits in the new order (must be 15 elements)
%    The example below simply reuses the old min/max values matched to 
%    each signature’s new position/order.
% -------------------------------------------------------------------------
yrange_min = [ ...
    0,   % sig_TotalRR  (was 0)
    0,   % sig_EventRR  (was 0)
    80,  % sig_HFD_mean (was 80)
  
    0,   % sig_x_percentile_95per (Q95)
    0,   % sig_x_Q_frequency_high (HF Freq)
    0,   % sig_x_Q_duration_high (HF Dur)
   
    0,   % sig_x_percentile_5per (Q5)
    0,   % sig_x_Q_frequency_low (LF Freq)
    0,   % sig_x_Q_duration_low (LF Dur)


    0,   % sig_BFI
    0,   % sig_BaseflowRecessionK (BFRC)
    -25,  % sig_FDC_slope (was -25)

    0,   % sig_FlashinessIndex
    0,  % sig_VariabilityIndex
    0];   % sig_RisingLimbDensity

yrange_max = [ ...
    1,    % sig_TotalRR
    1,    % sig_EventRR
    320,  % sig_HFD_mean

     10,   % sig_x_percentile_95per (Q95)
    0.5,    % sig_x_Q_frequency_high (HF Freq)
    80,   % sig_x_Q_duration_high (HF Dur)
    
    1.5,  % sig_x_percentile_5per (Q5)
    1,    % sig_x_Q_frequency_low (LF Freq)
    80,   % sig_x_Q_duration_low (LF Dur)




    1,    % sig_BFI
    0.7,  % sig_BaseflowRecessionK (BFRC)

    0,    % sig_FDC_slope

    1,    % sig_FlashinessIndex
    1.5, % sig_VariabilityIndex
    1];    % sig_RisingLimbDensity

% -------------------------------------------------------------------------
% 4) Define colors and other settings
% -------------------------------------------------------------------------
box_color      = [0.7, 0.7, 0.7];   % Grey for violin fill
observed_color = [0, 0, 0];         % Black for observed lines
observed_style = '--';              % Dashed line
% (Ensure 'colors' and 'OF_Plot' are defined as in your original script)

% -------------------------------------------------------------------------
% 5) Create figure with a 5x3 tiled layout
% -------------------------------------------------------------------------
%h = figure('units','normalized','outerposition',[0 0 0.7 1]);
%tiledlayout(5,3,'TileSpacing','compact');

h = figure('Units','normalized','OuterPosition',[0 0 0.6 1]);
t = tiledlayout(h,5,3,'TileSpacing','compact','Padding','tight');

% Loop over each signature
for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    disp(signature);
    
    % Go to the next tile in a 5x3 layout
    nexttile
    
    % Gather data for violin plots across objective functions (k = 1 to 8)
    all_data     = [];
    group_labels = [];
    
    for k = 1:8
        plot_data   = squeeze(Signatures_Array(i,k,:,:));
        nan_columns = all(isnan(plot_data), 1);
        plot_data(:, nan_columns) = NaN;
        
        % Append data and group labels
        for loc = 1:size(plot_data, 2)
            if isempty(all_data)
                all_data = plot_data(:, loc);
            else
                all_data = [all_data; plot_data(:, loc)];
            end
            group_labels = [group_labels; loc * ones(size(plot_data(:, loc)))];
        end
    end
    
    % Plot violin plot (simulated data)
    violinplot(all_data, group_labels, ...
               'ViolinColor', box_color, ...
               'ShowMean', false, ...
               'ShowData', false);
    hold on;
    
    % Overlay colorful circles for model medians
    for k = 1:8
        model_medians = median(squeeze(Signatures_Array(i,k,:,:)), 1, 'omitnan');
        scatter(1:length(model_medians), model_medians, 60, colors(k,:), 'filled');
        hold on;
    end
    
    % Overlay observed values as horizontal black lines
    for j = 1:length(catchments_aridity)
        catchment      = catchments_aridity{j};
        observed_value = obs_signatures_cali_bench.(signature).(catchment);
        if ~isnan(observed_value)
            line([j - 0.4, j + 0.4], [observed_value, observed_value], ...
                'Color', observed_color, 'LineWidth', 2, 'LineStyle', '-');
        end
    end
    
    % Set y-axis limits according to the new order
    ylim([yrange_min(i) yrange_max(i)]);
    title(label_signatures_new{i});
    grid minor;
    
    % Label the x-axis only on the bottom row (i.e., tiles 13,14,15)
    % ... after you set xlim/x ticks ...
    if i > 12
        xlabel('Catchments');
        xticklabels(catchments_labels);
        xtickangle(45);                   % reduces collisions
    else
        xticklabels([]);                  % hide labels on non-bottom rows
    end
    
    % Label the y-axis only in the left-most column (i.e., tiles 1,4,7,10,13)
    if mod(i,3) == 1
        ylabel('Signature Value');
    end
    
    % Set the x-ticks and labels for all plots
    xticks(1:length(catchments_aridity));
    xticklabels(catchments_labels);
    xlim([0.5 length(catchments_aridity) + 0.5]);
    
    % Adjust font size (assuming 'fontsize' is a custom function)
    fontsize(h, 10, "points");
end

% -------------------------------------------------------------------------
% 6) Create a legend outside the plot
% -------------------------------------------------------------------------
% Create dummy scatter and line objects for the legend *after* the loop.
figure(h);  % Ensure we’re referencing the correct figure
hold on;

% Dummy scatter plots for each objective-function color
h_leg_objs = gobjects(1,8);
for k = 1:8
    h_leg_objs(k) = scatter(nan, nan, 60, colors(k,:), 'filled');
end

% Dummy line for observed
h_leg_obs = plot(nan, nan, observed_style, 'Color', observed_color, 'LineWidth', 2);

% Combine into one array for the legend
legend_handles = [h_leg_objs, h_leg_obs];
legend_labels  = [OF_Plot, {'Observed'}];

% Create the legend
lgd = legend(legend_handles, legend_labels, ...
    'Location', 'south', ...          % We'll manually position it
    'Orientation', 'horizontal', ... % Keep it in one horizontal line
    'Box', 'off', ...
    'FontSize', 15);
%'NumColumns', 1, ...  % Forces a single row
    

% Manually place the legend at the bottom, centered across the figure.
% Adjust these values until the legend is positioned as you prefer.
%lgd.Units = 'normalized';
%lgd.Position = [0.05, 0.01, 0.8, 0.05];


filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Signature_Comp_5x3.jpg']);
exportgraphics(h,filename,'Resolution',300)

%% Plot Comparison of Two Objective Functions
colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

h=figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

%label_signatures_new = {'Total RR (-)','5th SF Perc (mm/d)','LF Dur (days)','LF Freq (-)'...
%    '95th SF Perc (mm/d)','HF Dur (days)','HF Freq (-)','MHFD (DOY)','FDC Slope (-)','Event RR (-)','BFI (-)','BFRC (-)',...  
%    'Flashiness Index (-)','Variability Index (-)','Rising Limb Density (-)'};

%yrange_min = [0,0,  0, 0,  0 ,  0,0,  50, -30,0,0,0,0,0,0];
%yrange_max = [1,1.2,80,0.8,10 ,50,0.5,300, 0, 1,1,1,1,1,1];

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    nexttile
    for k = [1,4] %1:length(objective_functions)
        plot_data = squeeze(Signatures_Array(i,k,:,:));

        nan_columns = all(isnan(plot_data), 1);

        % Replace entirely NaN columns with a placeholder (e.g., 0 or a very small number)
        % This example uses 0 for simplicity. Adjust as needed for your visualization.
        plot_data(:, nan_columns) = -40;

        violinplot(plot_data,1,'ViolinColor',colors(k,:));
        if i == 1
            legend(OF_Plot{k},'location','southwestoutside')
        %legend('KGE','','','','log NSE','location','best')
        end
        hold on
        for j = 1:length(catchments_aridity)
            catchment = catchments_aridity{j};
            store(j)=obs_signatures_cali_bench.(signature).(catchment);
                
        end
        plot(1:length(catchments_aridity),store,"-x",'Color','k');
        %legend('KGE','log KGE')
    end
    ylim([yrange_min(i) yrange_max(i)])
    title(label_signatures_new{i})
    grid minor
    xticks()
    xticklabels(catchments_labels);

    fontsize(h,20,"points")
    xlabel('Catchments');
    ylabel('Signature Value');
    if i == 1
        %legend()
        %legend('KGE','','','','log NSE','location','best')
    end
end

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Signature_Comp_old.jpg']);
exportgraphics(h,filename,'Resolution',300)

%%
% Define colors for 8 objective functions
colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

% Create figure
h = figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

% Define Y-axis range limits for each signature
%yrange_min = [0,0,  0, 0,  0 ,  0,0,  80, -25,0,  0,  0,0,0,0];
%yrange_max = [1,1.2,80,1,10 ,40,0.4, 320,  0,  0.7,1,0.6,1.5,1.5,1];

% Names for the legend
OF_Plot = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    nexttile
    
    % Loop through all 8 objective functions
    for k = 1:8
        plot_data = squeeze(Signatures_Array(i,k,:,:));
        %plot_data = squeeze(Signature_norm(i,k,:,:));

        % Handle NaN values for proper visualization
        nan_columns = all(isnan(plot_data), 1);
        plot_data(:, nan_columns) = NaN;  % Leave NaN to avoid plotting them

        % Calculate median for each catchment
        median_values = median(plot_data, 1, 'omitnan');
        
        % Plot the median values for the current objective function with the appropriate color
        plot(1:length(median_values), median_values, '-o', 'Color', colors(k,:), 'LineWidth', 2, 'MarkerSize', 8)
        hold on
    end
    
    % Overlay observed signatures for the catchments
    store = nan(1, length(catchments_aridity));  % Initialize store array
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        store(j) = obs_signatures_cali_bench.(signature).(catchment);
    end
    plot(1:length(catchments_aridity), store, '-x', 'Color', 'k', 'LineWidth', 5);
    %yline(0, '-x', 'Color', 'red', 'LineWidth', 5);
    
    % Set y-axis limits
    ylim([yrange_min(i) yrange_max(i)])
    %ylim([-1 1])
    % Add title, adjust grid and labels
    title(label_signatures_new{i})
    grid minor
    
    % Only show x-axis label on the bottom row
    if i > 12
        xlabel('Catchments');
    end
    
    % Only show y-axis label on the left column
    if mod(i, 4) == 1
        ylabel('Signature Value');
    end
    
    % Set the x-ticks and labels for all plots
    xticks(1:length(catchments_aridity));
    xticklabels(catchments_labels);
    xlim([0.5 10.5])
    
    % Set font size for the figure
    fontsize(h,12,"points")
end

% Create an extra axis for the legend
legend_ax = axes('Position', [0.8, 0.1, 0.1, 0.1], 'Visible', 'off');
hold(legend_ax, 'on');

% Plot a dummy line and points for the legend
h_leg_objs = arrayfun(@(k) plot(nan, nan, 'o-', 'Color', colors(k,:), 'LineWidth', 2, 'MarkerSize', 8), 1:8);
h_leg_obs = plot(nan, nan, '-x', 'Color', 'k', 'LineWidth', 3); % Observed dummy

% Create the legend with the specified OF_Plot names
legend([h_leg_objs, h_leg_obs], ...
    [OF_Plot, {'Observed'}], 'Location', 'best', 'FontSize', 14);

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Signature_Comp_median_test.jpg']);
exportgraphics(h,filename,'Resolution',300)

%% Comparison Plot with underlying violin
% Define Y-axis range limits for each signature
yrange_min = [0, 0, 80,-25,  ...    % 1st row
              0, 0, 0, 0, ...     % 2nd row
              0, 0, 0, 0, ...   % 3rd row
              0, 0, 0];           % 4th row
              
yrange_max = [1, 1,  320,0, ...   % 1st row
              1.5, 80, 1, 1, ...     % 2nd row
              10, 80, 1, 0.7, ...   % 3rd row
              1, 1, 1.5];          % 4th row
% Define labels for signatures
%label_signatures_new = { ...
    % 'Total RR (-)', 'Event RR (-)',  'MHFD (DOY)', 'FDC Slope (-)',...             % 1st row
    % 'Q5 (mm/d)', 'LF Dur (days)', 'LF Freq (-)', 'BFI (-)', ...               % 2nd row
    % 'Q95 (mm/d)', 'HF Dur (days)', 'HF Freq (-)', 'BFRC (-)', ...             % 3rd row
    % 'Rising Limb Density (-)', 'Flashiness Index (-)', 'Variability Index (-)'}; % 4th row

% Define colors and settings
box_color = [0.7, 0.7, 0.7];  % Grey for violin fill
observed_color = [0, 0, 0];    % Black for observed
observed_style = '--';         % Dashed line

% Create figure
h = figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    disp(signature)
    nexttile
    
    % Gather data for violin plots
    all_data = [];    % Initialize matrix for all model values
    group_labels = []; % Grouping variable for violin plot
    for k = 1:8       % Loop through objective functions
        plot_data = squeeze(Signatures_Array(i,k,:,:));
        nan_columns = all(isnan(plot_data), 1);
        plot_data(:, nan_columns) = NaN;
        
        % Append data and group labels
        for loc = 1:size(plot_data, 2)
            if isempty(all_data)
                all_data = plot_data(:, loc);
            else
                all_data = [all_data; plot_data(:, loc)];
            end
            group_labels = [group_labels; loc * ones(size(plot_data(:, loc)))];
        end
    end
    
    % Plot violin plot for simulated data
    violinplot(all_data, group_labels, 'ViolinColor', box_color, 'ShowMean', false,'ShowData',false);
    hold on;

    % Overlay colorful circles for model medians
    for k = 1:8
        model_medians = median(squeeze(Signatures_Array(i,k,:,:)), 1, 'omitnan');
        scatter(1:length(model_medians), model_medians, 60, colors(k,:), 'filled');
        hold on;
    end
    
    % Overlay observed values as a black dashed line
    % Overlay observed values as horizontal lines
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        observed_value = obs_signatures_cali_bench.(signature).(catchment);
        if ~isnan(observed_value)
            % Draw a horizontal line at the observed value
            line([j - 0.4, j + 0.4], [observed_value, observed_value], 'Color', observed_color, ...
                 'LineWidth', 2, 'LineStyle', '-');  % Horizontal line
        end
    end
    
        % Set y-axis limits
    ylim([yrange_min(i) yrange_max(i)]);
    title(label_signatures_new{i});
    grid minor;

    % Only show x-axis label on the bottom row
    if i > 12
        xlabel('Catchments');
    end
    
    % Only show y-axis label on the left column
    if mod(i, 4) == 1
        ylabel('Signature Value');
    end
    
    % Set the x-ticks and labels for all plots
    xticks(1:length(catchments_aridity));
    xticklabels(catchments_labels);
    xlim([0.5 length(catchments_aridity) + 0.5]);
    
    % Set font size for the figure
    fontsize(h, 20, "points");
end

% Create a legend
legend_ax = axes('Position', [0.8, 0.1, 0.1, 0.1], 'Visible', 'off');
hold(legend_ax, 'on');

% Dummy plot for legend
h_leg_objs = arrayfun(@(k) scatter(nan, nan, 60, colors(k,:), 'filled'), 1:8);
h_leg_obs = plot(nan, nan, observed_style, 'Color', observed_color, 'LineWidth', 2);

% Add legend
legend([h_leg_objs, h_leg_obs], [OF_Plot, {'Observed'}], 'Location', 'best', 'FontSize', 14);

% Save the figure
filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Signature_Comp_violin.jpg']);
exportgraphics(h, filename, 'Resolution', 300);


%% SIGNATURE COMP WITH ADJUSTED LEGEND
h = figure('units', 'normalized', 'outerposition', [0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

%yrange_min = [0,0,0,50,-30,0,0,0,0];
%yrange_max = [1,1.5,10,300,0,1,1,1.5,1];

yrange_min = [0,0,  0, 0,  0 ,  0,0,  50, -30,0,0,0,0,0,  0];
yrange_max = [1,1.2,50,0.5,10 ,50,0.5,300, 0, 1,1,1,1,1.5,1];

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    nexttile
    
    hold on
    
    % Plot KGE (k=1) and log NSE (k=4)
    h_violin = cell(1, 2); % Cell array to store Violin plot objects
    violin_index = 1;  % Index to store violinplot handles
    
    for k = [1, 4] % Loop for KGE and log NSE
        plot_data = squeeze(Signatures_Array(i,k,:,:));
        nan_columns = all(isnan(plot_data), 1);
        plot_data(:, nan_columns) = -40; % Replace NaNs with placeholder
        
        % Plot the violin and store the patch handle
        violin_obj = violinplot(plot_data, 1, 'ViolinColor', colors(k,:));
        h_violin{violin_index} = violin_obj(1).ViolinPlot; % Extract the patch object from Violin
        violin_index = violin_index + 1;
    end
    
    % Plot the observed data as a line
    store = nan(1, length(catchments_aridity));  % Initialize storage for observed data
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        store(j) = obs_signatures_cali_bench.(signature).(catchment);
    end
    h_obs = plot(1:10, store, "-x", 'Color', 'k', 'LineWidth', 1.5); % Observed signature

    % Set y-axis limits
    ylim([yrange_min(i) yrange_max(i)])
    title(label_signatures_new{i})
    grid minor
    
    % Set x-axis labels
    xticks(1:10);
    xticklabels(catchments_labels);

    % Set fontsize and labels
    fontsize(h, 15, "points")
    xlabel('Catchments');
    ylabel('Signature Value');
    
    % Only create the legend for the first plot
    if i == 1
        % Use the patch objects of the violin plots for the legend, along with the line plot
        legend([h_violin{1}, h_violin{2}, h_obs], {'KGE', 'log NSE', 'Observed'}, 'Location', 'best')
    end
end

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Signature_Comp.jpg']);
exportgraphics(h, filename, 'Resolution', 300);

%%
h = figure('units', 'normalized', 'outerposition', [0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

% Define y-range limits
yrange_min = [0,0,  0, 0,  0 ,  0,0,  100, -30,0, 0,0,0,0,  0];
yrange_max = [1,1.2,70,0.8,10 ,30,0.5,330, 0, 0.8,1,1,1,1.5,1];

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    nexttile
    
    hold on
    
    % Plot KGE (k=1) and log NSE (k=4)
    h_violin = cell(1, 2); % Cell array to store Violin plot objects
    violin_index = 1;  % Index to store violinplot handles
    
    for k = [1, 4] % Loop for KGE and log NSE
        plot_data = squeeze(Signatures_Array(i,k,:,:));
        %plot_data = squeeze(Signature_norm(i,k,:,:));

        nan_columns = all(isnan(plot_data), 1);
        plot_data(:, nan_columns) = -40; % Replace NaNs with placeholder
        
        % Plot the violin and store the patch handle
        violin_obj = violinplot(plot_data, 1, 'ViolinColor', colors(k,:));
        h_violin{violin_index} = violin_obj(1).ViolinPlot; % Extract the patch object from Violin
        violin_index = violin_index + 1;
    end
    
    % Plot the observed data as a line
    store = nan(1, length(catchments_aridity));  % Initialize storage for observed data
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        store(j) = obs_signatures_cali_bench.(signature).(catchment);
    end
    %h_obs = plot(1:10, store, "-x", 'Color', 'k', 'LineWidth', 1.5); % Observed signature

    % Set y-axis limits
    ylim([yrange_min(i) yrange_max(i)])
    title(label_signatures_new{i})
    grid minor
    
    % Set x-axis labels only for the bottom row
    if i > 12 % Bottom row (plots 13–16)
        xlabel('Catchments');
    end
    
    % Set y-axis labels only for the left column
    if mod(i, 4) == 1 % First column (plots 1, 5, 9, 13)
        ylabel('Signature Value');
    end

    % Keep xticks and yticks visible
    xticks(1:10);
    xlim([0.5 10.5])
    xticklabels(catchments_labels);
    
    % Set font size
    fontsize(h, 15, "points")
end

% Create an extra axis for the legend
legend_ax = axes('Position', [0.8, 0.1, 0.1, 0.1], 'Visible', 'off');
hold(legend_ax, 'on');

% Plot a dummy line and violins for the legend
h_leg1 = plot(legend_ax, nan, nan, 'Color', colors(1,:), 'LineWidth', 2); % KGE dummy
h_leg2 = plot(legend_ax, nan, nan, 'Color', colors(4,:), 'LineWidth', 2); % log NSE dummy
h_leg_obs = plot(legend_ax, nan, nan, '-xk', 'LineWidth', 1.5); % Observed dummy

% Set up the legend in the new axis with a larger font size
legend(legend_ax, [h_leg1, h_leg2, h_leg_obs], {'KGE', 'log NSE', 'Observed'}, 'Location', 'best', 'FontSize', 20);

% Increase the size of legend markers
legend_ax.Children(1).MarkerSize = 10; % Increase marker size for observed data
legend_ax.Children(2).LineWidth = 3;   % Increase line width for log NSE
legend_ax.Children(3).LineWidth = 3;   % Increase line width for KGE

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Signature_Comp.jpg']);
exportgraphics(h, filename, 'Resolution', 300);

%% COMPARISON CALI AND EVAL

h=figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

%yrange_min = [0,0,0,50,-30,0,0,0,0];
%yrange_max = [1,1.5,10,300,0,1,1,1.5,1];

for i = 1:length(sorted_sig_new)
    signature = sorted_sig_new{i};
    nexttile
    for k = [1] %1:length(objective_functions)
        plot_data = squeeze(Signatures_Array_comp(i,k,:,:)*100);
        nan_columns = all(isnan(plot_data), 1);
        % Replace entirely NaN columns with a placeholder (e.g., 0 or a very small number)
        % This example uses 0 for simplicity. Adjust as needed for your visualization.
        plot_data(:, nan_columns) = -40;
        violinplot(plot_data,1,'ViolinColor',colors(k,:));
        hold on
        
        for j = 1:length(catchments_aridity)
            catchment = catchments_aridity{j};
            store(j)=obs_signatures_cali_bench.(signature).(catchment);
                
        end
        %plot(1:11,store,"-x",'Color','red');
        %legend('KGE','log KGE')
    end
    %ylim([yrange_min(i) yrange_max(i)])
    title(label_signatures_new{i})
    grid minor
    xticks()
    xticklabels(catchments_labels);
    fontsize(h,20,"points")
    xlabel('Catchments');
    ylabel('Relative Signature Change (%)');
    ylim([-50 50])
    xlim([0.5 10.5])
end

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/Cali_vs_Vali.jpg']);
exportgraphics(h,filename,'Resolution',300)

%% NORMALIZED OVERALL PLOT
f = figure('units', 'normalized', 'outerposition', [0 0 0.5 1]);
tiledlayout(4, 2, 'TileSpacing', 'Compact');

label_signatures_new_brief = { ...
    'TRR (-)', 'ERR (-)',  'MHFD (DOY)', ...             % 1st row
    'Q95 (mm/d)',  'HFF (-)','HFD (days)',  ...             % 3rd row
    'Q5 (mm/d)', 'LFF (-)','LFD (days)',   ...               % 2nd row
    'BFI (-)', 'BFRC (-)','FDC Slope (-)',... % 4th row
    'FI (-)', 'VI (-)', 'RLD (-)'}; % 5th row



for k = 1:8 % Loop over models (1 to 4 in this case)
    obj_fun = objective_functions{k};
    nexttile
    
    % Plot the violin plot
    h = violinplot(Error_Norm_Values_range(:, :, k)', 1, 'ViolinColor', colors(k, :));
    
    % Set y-axis limits
    ylim([-1 1])
    
    % Add the "Zero Error" line
    yline(0, '-', 'Zero Error', 'Color', 'red');
    
    % Add vertical labels "bad" at y = -1, "good" at y = 0, and "bad" at y = 1 on the right side
    %text(16.5, -1.125, 'bad', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black', 'Rotation', 90);
    %text(16.5, -0.125, 'good', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black', 'Rotation', 90);
    %text(16.5, 0.875, 'bad', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black', 'Rotation', 90);
      
    % Set x-ticks and x-tick labels
    xticks(1:15);
    xticklabels(label_signatures_new_brief);
    
    % Set the title for each plot
    title(OF_Plot{k});
    fontsize(f,8,"points")

    % Add a y-axis label only for the first plot
    if k == 1 || k == 3 || k == 5 || k == 7
        ylabel('Normalized Signature Error', 'FontSize', 12);
    end
end

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/violin_errors_all.jpg']);
exportgraphics(f, filename, 'Resolution', 300);

%% make a table with values
%% ==== Mean & Std per objective (rows) and signature (columns) ====
% Assumes:
%   Error_Norm_Values_range : 3D array with a 15-signature dimension and an 8-objective dimension.
%   label_signatures_new    : 1x15 cellstr with signature labels (column names)
%   objective_functions     : 1x8  cellstr with objective names (for row labels)

A = Error_Norm_Values_range;

% ---- Identify dimensions ----
sz = size(A);
sigDim = find(sz == 15, 1, 'first');
if isempty(sigDim)
    error('Could not find a dimension of size 15 (signatures).');
end
objDim = find(sz == 8, 1, 'first');
if isempty(objDim)
    % Fall back to 3rd dimension if not exactly 8 (adjust if needed)
    objDim = ndims(A);
    warning('Assuming objectives are along dimension %d.', objDim);
end

% ---- Bring array to [Nsample x Nsig(=15) x Nobj(=8)] ----
% Determine remaining dimension as "samples"
allDims = 1:ndims(A);
smpDim = setdiff(allDims, [sigDim objDim]);
if numel(smpDim) ~= 1
    error('Could not uniquely identify the samples dimension.');
end
Aperm = permute(A, [smpDim, sigDim, objDim]);   % [Nsample x 15 x 8]

% ---- Compute median and std across samples (dim=1) ----
medianSigObj = squeeze(median(Aperm, 1, 'omitnan'));  % [15 x 8]
stdSigObj    = squeeze(std(Aperm, 0, 1, 'omitnan'));  % [15 x 8]

% ---- Arrange rows as 2x per objective: mean row then sd row ----
% We want a 16x15 matrix: for each k (objective), rows [mean; sd], columns = signatures
Nobj = size(meanSigObj, 2);   % should be 8
Nsig = size(meanSigObj, 1);   % should be 15
outMat = nan(2*Nobj, Nsig);
rowNames = cell(2*Nobj,1);

for k = 1:Nobj 
    outMat(2*k-1, :) = medianSigObj(:, k).';  % median row for objective k
    outMat(2*k,   :) = stdSigObj(:,  k).';    % sd row for objective k
    
    % Row names
    ok = objective_functions{k};
    rowNames{2*k-1} = sprintf('%s median', ok);
    rowNames{2*k}   = sprintf('%s sd',     ok);
end

% ---- Build table with signature names as columns ----
if exist('label_signatures_new','var') && numel(label_signatures_new) == Nsig
    varNames = matlab.lang.makeValidName(label_signatures_new, 'ReplacementStyle','delete');
else
    varNames = compose('Sig%d', 1:Nsig);
end
T_stats = array2table(outMat, 'VariableNames', varNames, 'RowNames', rowNames);

% ---- Show a preview (or disp the whole thing) ----
disp(T_stats);

% ---- (Optional) write to CSV for inspection) ----
% writetable resets RowNames, so include them as a column:
writetable(T_stats, 'violin_errors_stats.csv', 'WriteRowNames', true);

%%
% Repeat for the next set of objective functions
f = figure('units', 'normalized', 'outerposition', [0 0 0.5 0.5]);
tiledlayout(2, 2, 'TileSpacing', 'Compact');

for k = 5:length(objective_functions) % Loop over the remaining models (5 onwards)
    obj_fun = objective_functions{k};
    nexttile
    
    % Plot the violin plot
    h = violinplot(Error_Norm_Values_range(:, :, k)', 1, 'ViolinColor', colors(k, :));
    
    % Set y-axis limits
    ylim([-1 1])
    
    % Add the "Zero Error" line
    yline(0, '-', 'Zero Error', 'Color', 'red');
    
    % Add vertical labels "bad" at y = -1, "good" at y = 0, and "bad" at y = 1 on the right side
    text(16.5, -1.125, 'bad', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black', 'Rotation', 90);
    text(16.5, -0.125, 'good', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black', 'Rotation', 90);
    text(16.5, 0.875, 'bad', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'black', 'Rotation', 90);
    
    % Set x-ticks and x-tick labels
    xticks(1:15);
    xticklabels(label_signatures_new);
    
    % Set the title for each plot
    title(OF_Plot{k});
    
    % Add a y-axis label only for the first plot
    if k == 5 || k == 7
        ylabel('Normalized Signature Error', 'FontSize', 12);
    end
end

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/violin_errors_2.jpg']);
exportgraphics(f, filename, 'Resolution', 300);



%% Plot Range Limited Error Values
label_signatures_new2 = {'Total RR','Q5',...
    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
    'Flashiness Index','Rising Limb Density'};

catchments_labels = {'AUS1','BR6','AUS6','C12','C02','GB3','C02','HYS','GB2','GB8','LAM'};

label_signatures_new = {'Total RR','Q5','LF Dur','LF Freq'...
    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
    'FI','VI','RLD','StdDev'};

colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

%(5,101,23)

%cmap_pos = brewermap(8,"-RdYlGn");
%cmap_neg = brewermap(9,"PRGn");
%cmap_neg = flipud(cmap_pos);
%cmap = colormap(greenCenteredColormap(256));
%colorbar;
%cat(1,cmap_neg,cmap_pos); 
cmap = brewermap(19,"RdBu");
%colorbar
%cmap = [cmap_neg; cmap_pos];

f=figure('units','normalized','outerposition',[0 0 0.5 0.5]);
tiledlayout(2,2,'TileSpacing','Compact')

for k = 1:4%length(objective_functions) % number of models (currently: 3)
    obj_fun = objective_functions{k};
    nexttile
    %figure('units','normalized','outerposition',[0 0 0.5 0.5])
    %tiledlayout(4,2,'TileSpacing','Compact')
    %for j = 1:length(catchments_aridity)+1
        %catchment = catchments_aridity{j};
        %h = heatmap(Rank_values(:,:,k)','Colormap',brewermap(8,"-RdYlGn"),'CellLabelColor','none');
        %h = heatmap(Rank_values(:,:,k)','Colormap',brewermap(8,"-RdYlGn"),'CellLabelColor','none');
        %h = heatmap(Error_Norm_Values_range(:,:,k)','Colormap',cmap,'CellLabelColor','none');
        h = violinplot(Error_Norm_Values_range(:,:,k)',1,'ViolinColor',colors(k,:));
        %if k > 6 % Left column
        %h.XDisplayLabels = label_signatures_new2;
        %end
        %if k < 3 % Top row
        %h.YDisplayLabels = catchments_labels;
        %end
        clim([-1 1])
        S = struct(h); % Undocumented
        %ax = S.Axes; 
        %h.GridVisible = 'off';

        %xline(ax,[15.5,16.5],'k-','LineWidth',3);
        %yline(ax,[11.5,12.5],'k-','LineWidth',3);
                

        % Get the current data and the size of the heatmap
        %heatData = Error_Norm_Values_range(:,:,k)';
        %[numRows, numCols] = size(heatData);
        ylim([-1 1])
        yline(0,'-','Zero Error','Color','red')

        % Prepare custom labels
        %customLabels = repmat({''}, numRows, numCols); % Initialize with empty strings
        xticks(1:9);
        xticklabels(label_signatures_new2);
        % Fill in labels only for the last row and column
        %for row = 1:numRows
        %    customLabels{row, numCols} = sprintf('%.1f', heatData(row, numCols));
        %end
        %for col = 1:numCols
        %    customLabels{numRows, col} = sprintf('%.1f', heatData(numRows, col));
        %end
        
        % Manually set the cell labels
        %for row = 1:numRows
        %    for col = 1:numCols
        %        if ~isempty(customLabels{row, col})
                    %text(ax, col, row, customLabels{row, col}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle','FontSize', 15);
        %        end
        %    end
        %end
        %h.FontSize = 15;

    %end
    title(OF_Plot{k})
    %filename = sprintf('OF_Plot%d.png', k);
    %saveas(h, filename);
end
%sgtitle('Rank of Signature Representation (compared between OFs)')
filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/violin_errors.jpg']);
exportgraphics(f,filename,'Resolution',300)

f=figure('units','normalized','outerposition',[0 0 0.5 0.5]);
tiledlayout(2,2,'TileSpacing','Compact')

for k = 5:length(objective_functions) % number of models (currently: 3)
    obj_fun = objective_functions{k};
    nexttile
    %figure('units','normalized','outerposition',[0 0 0.5 0.5])
    %tiledlayout(4,2,'TileSpacing','Compact')
    %for j = 1:length(catchments_aridity)+1
        %catchment = catchments_aridity{j};
        %h = heatmap(Rank_values(:,:,k)','Colormap',brewermap(8,"-RdYlGn"),'CellLabelColor','none');
        %h = heatmap(Rank_values(:,:,k)','Colormap',brewermap(8,"-RdYlGn"),'CellLabelColor','none');
        %h = heatmap(Error_Norm_Values_range(:,:,k)','Colormap',cmap,'CellLabelColor','none');
        h = violinplot(Error_Norm_Values_range(:,:,k)',1,'ViolinColor',colors(k,:));
        %if k > 6 % Left column
        %h.XDisplayLabels = label_signatures_new2;
        %end
        %if k < 3 % Top row
        %h.YDisplayLabels = catchments_labels;
        %end
        %clim([-1 1])
        %S = struct(h); % Undocumented
        %ax = S.Axes; 
        %h.GridVisible = 'off';
        ylim([-1 1])
        yline(0,'-','Zero Error','Color','red')
        % Prepare custom labels
        %customLabels = repmat({''}, numRows, numCols); % Initialize with empty strings
        xticks(1:9);
        xticklabels(label_signatures_new2);
        %xline(ax,[15.5,16.5],'k-','LineWidth',3);
        %yline(ax,[11.5,12.5],'k-','LineWidth',3);
               
        % Get the current data and the size of the heatmap
        %heatData = Error_Norm_Values_range(:,:,k)';
        %[numRows, numCols] = size(heatData);
        
        % Prepare custom labels
        %customLabels = repmat({''}, numRows, numCols); % Initialize with empty strings
        
        % Fill in labels only for the last row and column
        %for row = 1:numRows
        %    customLabels{row, numCols} = sprintf('%.1f', heatData(row, numCols));
        %end
        %for col = 1:numCols
        %    customLabels{numRows, col} = sprintf('%.1f', heatData(numRows, col));
        %end
        
        % Manually set the cell labels
        %for row = 1:numRows
        %    for col = 1:numCols
        %        if ~isempty(customLabels{row, col})
        %            text(ax, col, row, customLabels{row, col}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle','FontSize', 15);
        %        end
        %    end
        %end
        %h.FontSize = 15;

    %end
    title(OF_Plot{k})
    %filename = sprintf('OF_Plot%d.png', k);
    %saveas(h, filename);
end
%sgtitle('Rank of Signature Representation (compared between OFs)')
filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/violin_errors_2.jpg']);
exportgraphics(f,filename,'Resolution',300)

%% LINE PLOT for ERROR METRIC
%label_signatures_new = {'Total RR','5th SF Perc','LF Dur','LF Freq'...
%    '95th SF Perc','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
%    'Flashiness Index','Variability Index','Rising Limb Density'};

label_signatures_new = {'','Total RR','Q5','LF Dur','LF Freq'...
    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
    'FI','VI','RLD'};

label_signatures_new2 = {'','Total RR','Q5',...
    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
    'Flashiness Index','Rising Limb Density'};

median_catchment_error_range(1,:)=0;
mean_catchment_error_range(1,:)=0;
f=figure('units','normalized','outerposition',[0 0 0.5 1]);
for k = 1:length(objective_functions)
    plot(cumsum(median_catchment_error_range(:,k')),'Color',colors(k,:),"LineWidth",10)
    hold on

end
%ylim([0 0.4])
xticks(1:10)
xticklabels(label_signatures_new2)
legend(OF_Plot,"Location",'northwest')
xlim([0.5 10.5])
grid on
fontsize(f,20,"points")

filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/lineplot_new_error.jpg']);
exportgraphics(f,filename,'Resolution',300)


%% Updated Violinplots like at the beginning
h=figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing', 'compact')

%label_signatures_violin = {'Total RR','Q5',...
%    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
%   'Flashiness Index','Rising Limb Density'};
label_signatures_new_nounit = { ...
    'Total RR', 'Event RR',  'MHFD', 'FDC Slope',...             % 1st row
    'Q5', 'LF Dur', 'LF Freq', 'BFI', ...               % 2nd row
    'Q95', 'HF Dur', 'HF Freq', 'BFRC', ...             % 3rd row
    'Rising Limb Density', 'Flashiness Index', 'Variability Index'}; % 4th row

%colors=parula(11);
store_median_errors = NaN(11,8,15);
store_obs = NaN(11,15);
for j = 1:length(sorted_sig_new)
    nexttile

    %store_median_errors(:,:,j) = store_median_errors(:,:,j)./variance_value;
    violinplot(squeeze(Error_Norm_Values_range(j,:,:)),1,'ViolinColor',colors(:,:));
    %boxplot(squeeze(Error_Norm_Values_range(j,:,:)));%,'ViolinColor',colors(:,:));
    plot(median(squeeze(Error_Norm_Values_range(j,:,:)),'omitnan'),"k")

    %boxplot(store_median_errors(:,:,j)./variance_value,'whisker',Inf);
    %plot(store_median_errors(xyz,:,j)'./variance_value,'.-','MarkerSize',30,'Color', colors(xyz,:));

    ylabel('Normalized Signature Error')
    title(strrep(convertCharsToStrings(label_signatures_new_nounit{j}),'_','\_'))
    xlim([0.5 8.5])

    xticks(1:8);
    xticklabels(OF_Plot);

    ylim([-1 1])

    %YL = get(gca, 'YLim');
    %maxlim = max(abs(YL));
    %set(gca, 'YLim', [-maxlim maxlim]);

    %set(gca, 'YScale', 'log')
    grid on
    %end

    if j==9
        %legend(strrep(catchments_aridity,'_','\_'),'location','northeastoutside')
    end
end

f = gcf;
fontsize(f,20,"points")
sgtitle('Normalized Signature Error','FontSize',30)
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/new_violinplot.jpg');
exportgraphics(f,filename,'Resolution',300)

%% RECALC OVER ALL 15 SIGNATURES
% Numeric Error 
Impact_Signatures = NaN(length(sorted_signatures),4);
Error_Values = NaN(15,8,47,10);
Norm_Values = NaN(15,1);
Error_Norm_Values = NaN(15,10,8);
Error_Norm_Values_range = NaN(9,10,8);
Signature_cal_eval = NaN(9,10,8);
Signature_cal_median = NaN(9,10,8);
median_catchment_error = NaN(15,8);
median_catchment_error_range = NaN(8,8);
mean_catchment_error_range = NaN(8,8);
norm_plot = NaN(15,10,2);
Rank_values = NaN(15+1,10+1,8);
Error_Values = NaN(15,8,47,10);
Signatures_Array = NaN(15,8,47,10);
Signatures_Array_comp = NaN(15,8,47,10);
Signatures_Array_vali = NaN(15,8,47,10);
Signature_eval_median = NaN(9,10,8);

%sorted_sig_new = {'sig_TotalRR','sig_x_percentile_5per','sig_x_Q_duration_low','sig_x_Q_frequency_low',...
%    'sig_x_percentile_95per','sig_x_Q_duration_high','sig_x_Q_frequency_high','sig_HFD_mean','sig_FDC_slope'...
%    'sig_EventRR','sig_BFI','sig_BaseflowRecessionK',...
%    'sig_FlashinessIndex','sig_VariabilityIndex','sig_RisingLimbDensity'};

%label_signatures_new = {'Total RR','Q5','LF Dur','LF Freq'...
%    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
%    'Flashiness Index','Variability Index','Rising Limb Density'};

sorted_sig_new2 = {'sig_TotalRR','sig_x_percentile_5per',...
    'sig_x_percentile_95per','sig_HFD_mean','sig_FDC_slope'...
    'sig_EventRR','sig_BFI',...
    'sig_FlashinessIndex','sig_RisingLimbDensity'};

label_signatures_new2 = {'Total RR','Q5',...
    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
    'Flashiness Index','Rising Limb Density'};

sorted_sig_new_significant = {'sig_TotalRR','sig_x_percentile_5per','sig_x_Q_frequency_low',...
    'sig_x_percentile_95per'...
    'sig_EventRR','sig_BFI','sig_BaseflowRecessionK',...
    'sig_FlashinessIndex'};

label_signatures_new_significant = {'Total RR','Q5','LF Freq'...
    'Q95','Event RR','BFI','BFRC',...  
    'Flashiness Index'};


for i = 1:length(sorted_sig_new_significant)
    signature = sorted_sig_new_significant{i};
    norm_values_store = NaN(10,1);
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        for k = 1:length(objective_functions) % number of models (currently: 3)
            %model = model_list_sorted{i};
            obj_fun = objective_functions{k};
            for l = 1:numel(model_list)
                model = model_list{l};
                try 
                    Error_Values(i,k,l,j) = sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment);
                    Signatures_Array(i,k,l,j) = sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun);
                    Signatures_Array_vali(i,k,l,j) = sim_signatures_eval_bench.(signature).(catchment).(model).(obj_fun);%-obs_signatures_cali_bench.(signature).(catchment);
                    Signatures_Array_comp(i,k,l,j) = Signatures_Array(i,k,l,j)- Signatures_Array_vali(i,k,l,j);

                    %if obs_signatures_cali_bench.(signature).(catchment) ~= 0
                            
                        %/obs_signatures_cali_bench.(signature).(catchment);
                    %else
                    %end
                catch
                end
            end
        end
        norm_values_store(j) = obs_signatures_cali_bench.(signature).(catchment);
    end
    Norm_Values(i) = max(norm_values_store)-min(norm_values_store);
    for j=1:10
        for k=1:8
            
            %Error_Norm_Values(i,j,k) = median(Error_Values(i,k,:,j),3,'omitnan')./Norm_Values(i);
            % New normalization using the maximum error
            Error_Norm_Values(i,j,k) = median(Error_Values(i,k,:,j),3,'omitnan');
            Signature_cal_median(i,j,k) = median(Signatures_Array(i,k,:,j),3,'omitnan');
            Signature_cal_eval(i,j,k) = median((Signatures_Array(i,k,:,j)-Signatures_Array_vali(i,k,:,j)),3,'omitnan');
            %median_catchment_error(i,k) = median(Error_Norm_Values(i,:,k),"all",'omitnan');
        end

        %norm_plot(i,j,1) = min(Error_Norm_Values(i,j,:),[],[3],'omitnan');
        norm_plot(i,j,2) = max(abs(Error_Norm_Values(i,j,:)),[],[3],'omitnan');
    end


    for j=1:10
        for k=1:8
            %Error_Norm_Values_range(i,j,k) = (Error_Norm_Values(i,j,k) - norm_plot(i,1))/(norm_plot(i,2)-norm_plot(i,1));
            Error_Norm_Values_range(i,j,k) = (Error_Norm_Values(i,j,k))/(norm_plot(i,j,2));

            median_catchment_error_range(i,k) = median(abs(Error_Norm_Values_range(i,:,k)),"all",'omitnan');
            mean_catchment_error_range(i,k) = mean(abs(Error_Norm_Values_range(i,:,k)),"all",'omitnan');

        end
        % Rank the Error_Norm_Values for each signature across all objective functions
        
        %[~, nonnanrank] = sort(A(validmask));
        %ranks = NaN(size(A));
        %rank(validmask) = nonnanrank;

        validmask = ~isnan(Error_Norm_Values(i,j,:));
        sort_array = Error_Norm_Values(i,j,:);
        [~, ranked_indices] = sort(sort_array(validmask)); % Sort to get the ranking
        ranks_1 = NaN(size(Error_Norm_Values(i,j,:)));
        ranks_1(validmask) = ranked_indices;
        
        validmask_2 = ~isnan(ranks_1);
        [~, ranks] = sort(ranks_1(validmask_2)); % Get the ranks (1 for lowest value)
        ranks_2 = NaN(size(Error_Norm_Values(i,j,:)));
        ranks_2(validmask_2) = ranks;

        Rank_values(i,j,1:length(ranks_2)) = ranks_2; % Store the ranks

        for k = 1:8

            Rank_values(10,j,k) = mean(Rank_values(1:9,j,k),'omitnan');
            Rank_values(i,12,k) = mean(Rank_values(i,1:11,k),'omitnan');
            Rank_values(10,12,k) = NaN;

            
            %Error_Norm_Values_range(10,j,k) = std(Error_Norm_Values_range(1:9,j,k),'omitnan');
            %Error_Norm_Values_range(i,12,k) = std(Error_Norm_Values_range(i,1:11,k),'omitnan');
            %Error_Norm_Values_range(10,12,k) = NaN;
        end
    end

end

%% STACKED BAR PLOT VERSION
% Define the signatures and objective functions (unchanged)
label_signatures_new = {'','Total RR','Q5','LF Dur','LF Freq'...
    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
    'FI','VI','RLD'};

label_signatures_new2 = {'','Total RR','Q5',...
    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
    'Flashiness Index','Rising Limb Density'};

% Example data for median catchment error range (modify with actual data)
%median_catchment_error_range = rand(length(objective_functions), 10); % Example random data (sorted by objective function)

% Create a gradient of blue colors from dark to light
num_signatures = size(median_catchment_error_range, 1);  % Number of signatures
%blue_gradient = flipud([linspace(0.1, 0.8, num_signatures)', linspace(0.1, 0.9, num_signatures)', ones(num_signatures, 1)]); % Dark to light blue gradient
% Create a red gradient from dark red to light red
graydient = flipud( repmat( linspace(0.2, 0.9, num_signatures)', 1, 3) );
% Create a new figure
f = figure('units', 'normalized', 'outerposition', [0 0 0.5 0.7]);

% Create a horizontal stacked bar plot where each group represents an objective function
bh = barh(mean_catchment_error_range', 'stacked', 'LineWidth', 1.5, FaceColor=([0.25 0.25 0.25]));

% Set the colors for each signature using the blue gradient
for k = 1:num_signatures
    bh(k).FaceColor = graydient(k,:);
end

% Customize the axes and labels
yticks(1:length(objective_functions));
yticklabels(OF_Plot);  % Objective functions are the y-tick labels
legend(label_signatures_new_significant, "Location", 'northeast');  % Legend represents the signatures
xlim([0 7]);  % Adjust the x-limits depending on your data
grid on;
set(gca, 'YDir', 'reverse')
% Set font size
fontsize(f, 20, "points");
ylabel("Objective Functions")
xlabel("Cumulative Normalized Signature Error")
% Export the plot as a high-resolution image
filename = sprintf(['/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/stacked_barplot_error.jpg']);
exportgraphics(f, filename, 'Resolution', 300);


%% RERUN ERROR CALCULATION FOR ALL SIGNATURES

Impact_Signatures = NaN(length(sorted_signatures),4);
Error_Values = NaN(15,8,47,10);
Norm_Values = NaN(15,1);
Error_Norm_Values = NaN(15,10,8);
median_catchment_error = NaN(15,8);
median_catchment_error_range = NaN(10,8);
mean_catchment_error_range = NaN(10,8);
norm_plot = NaN(15,10,2);
Rank_values = NaN(15+1,10+1,8);

%sorted_sig_new = {'sig_TotalRR','sig_x_percentile_5per','sig_x_Q_duration_low','sig_x_Q_frequency_low',...
%    'sig_x_percentile_95per','sig_x_Q_duration_high','sig_x_Q_frequency_high','sig_HFD_mean','sig_FDC_slope'...
%    'sig_EventRR','sig_BFI','sig_BaseflowRecessionK',...
%    'sig_FlashinessIndex','sig_VariabilityIndex','sig_RisingLimbDensity'};

%label_signatures_new = {'Total RR','Q5','LF Dur','LF Freq'...
%    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
%    'Flashiness Index','Variability Index','Rising Limb Density'};

sorted_sig_new2 = {'sig_TotalRR','sig_x_percentile_5per',...
    'sig_x_percentile_95per','sig_HFD_mean','sig_FDC_slope'...
    'sig_EventRR','sig_BFI',...
    'sig_FlashinessIndex','sig_RisingLimbDensity'};

label_signatures_new2 = {'Total RR','Q5',...
    'Q95','MHFD','FDC Slope','Event RR','BFI',...  
    'Flashiness Index','Rising Limb Density'};
%% 
sorted_sig_new = { ...
    'sig_TotalRR', 'sig_EventRR',  'sig_HFD_mean', 'sig_FDC_slope',...  % 1st row
    'sig_x_percentile_5per', 'sig_x_Q_duration_low', 'sig_x_Q_frequency_low', 'sig_BFI', ... % 2nd row
    'sig_x_percentile_95per', 'sig_x_Q_duration_high', 'sig_x_Q_frequency_high', 'sig_BaseflowRecessionK', ... % 3rd row
    'sig_RisingLimbDensity', 'sig_FlashinessIndex', 'sig_VariabilityIndex'}; % 4th row


sig_corr_plot = {'sig_TotalRR', 'sig_EventRR','sig_x_percentile_95per','sig_x_percentile_5per'...
    'sig_BFI', 'sig_BaseflowRecessionK', 'sig_FlashinessIndex','sig_x_Q_frequency_low', ...
    'sig_x_Q_frequency_high', 'sig_FDC_slope', 'sig_VariabilityIndex', 'sig_x_Q_duration_low',...
     'sig_x_Q_duration_high', 'sig_HFD_mean','sig_RisingLimbDensity',
    };
% -------------------------------------------------------------------------
% 2) Define user-friendly labels in the same new order
% -------------------------------------------------------------------------
label_signatures_new = { ...
    'Total RR (-)', ...
    'Event RR (-)', ...
    'MHFD (DOY)', ...
    'FDC Slope (-)', ...
    'Q5 (mm/d)', ...
    'LF Dur (days)', ...
    'LF Freq (-)', ...
    'BFI (-)', ...
    'Q95 (mm/d)', ...
    'HF Dur (days)', ...
    'HF Freq (-)', ...
    'BFRC (-)', ...
    'Rising Limb Density (-)', ...
    'Flashiness Index (-)', ...
    'Variability Index (-)'};

label_signatures_corr_plot = { ...
    'Total RR (-)', ...             % sig_TotalRR
    'Event RR (-)', ...             % sig_EventRR
    'Q95 (mm/d)', ...               % sig_x_percentile_95per
    'Q5 (mm/d)', ...                % sig_x_percentile_5per
    'BFI (-)', ...                  % sig_BFI
    'BFRC (-)', ...                 % sig_BaseflowRecessionK
    'Flashiness Index (-)', ...     % sig_FlashinessIndex
    'LF Freq (-)', ...              % sig_x_Q_frequency_low
    'HF Freq (-)', ...              % sig_x_Q_frequency_high
    'FDC Slope (-)', ...            % sig_FDC_slope
    'Variability Index (-)', ...    % sig_VariabilityIndex
    'LF Dur (days)', ...            % sig_x_Q_duration_low
    'HF Dur (days)', ...            % sig_x_Q_duration_high
    'MHFD (DOY)', ...               % sig_HFD_mean
    'Rising Limb Density (-)'};     % sig_RisingLimbDensity

obs_values = NaN(15,10);

for i = 1:length(sig_corr_plot)
    signature = sig_corr_plot{i};
    norm_values_store = NaN(10,1);
    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        obs_values(i,j) = obs_signatures_cali_bench.(signature).(catchment);
        for k = 1:length(objective_functions) % number of models (currently: 3)
            %model = model_list_sorted{i};
            obj_fun = objective_functions{k};
            for l = 1:numel(model_list)
                model = model_list{l};
                try 
                    Error_Values(i,k,l,j) = sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun);%-obs_signatures_cali_bench.(signature).(catchment);
                    
                    %if obs_signatures_cali_bench.(signature).(catchment) ~= 0
                            
                        %/obs_signatures_cali_bench.(signature).(catchment);
                    %else
                    %end
                catch
                end
            end
        end
        norm_values_store(j) = obs_signatures_cali_bench.(signature).(catchment);
    end
    Norm_Values(i) = max(norm_values_store)-min(norm_values_store);
    for j=1:10
        for k=1:8
            
            %Error_Norm_Values(i,j,k) = median(Error_Values(i,k,:,j),3,'omitnan')./Norm_Values(i);
            % New normalization using the maximum error
            Error_Norm_Values(i,j,k) = median(Error_Values(i,k,:,j),3,'omitnan');
            %median_catchment_error(i,k) = median(Error_Norm_Values(i,:,k),"all",'omitnan');
        end

        %norm_plot(i,j,1) = min(Error_Norm_Values(i,j,:),[],[3],'omitnan');
        norm_plot(i,j,2) = max(Error_Norm_Values(i,j,:),[],[3],'omitnan');
    end


    for j=1:10
        for k=1:8
            %Error_Norm_Values_range(i,j,k) = (Error_Norm_Values(i,j,k) - norm_plot(i,1))/(norm_plot(i,2)-norm_plot(i,1));
            Error_Norm_Values_range(i,j,k) = (Error_Norm_Values(i,j,k))/(norm_plot(i,j,2));

            median_catchment_error_range(i+1,k) = median(Error_Norm_Values_range(i,:,k),"all",'omitnan');
            %mean_catchment_error_range(i+1,k) = mean(Error_Norm_Values_range(i,:,k),"all",'omitnan');

        end
        % Rank the Error_Norm_Values for each signature across all objective functions
        
    end

end

%% DO STATISTICAL TESTING
%label_signatures_new = {'Total RR','Q5','LF Dur','LF Freq'...
%    'Q95','HF Dur','HF Freq','MHFD','FDC Slope','Event RR','BFI','BFRC',...  
%    'Flashiness Index','Variability Index','Rising Limb Density'};

save_p_value = NaN(15,28);

for i = 1:length(sig_corr_plot)
    counter = 1;

    for k = 1:length(objective_functions) % number of models (currently: 3)
        for j = 1:length(objective_functions)
            if k == j || k>j
            else
                % Get value for the the combination of signature and objective
                % function

                x = reshape(Error_Norm_Values(i,:,k),[],1);
                
                y = reshape(Error_Norm_Values(i,:,j),[],1);
                disp(x-y)
                %disp(y)

                [h,p,ci,stats] = ttest(x,y);
                %disp(h)
                disp(p)
                disp(ci)
                a = nanmean(x)-nanmean(y);
                %disp(a)
                b = sqrt((nanstd(x)*nanstd(x)/10)+(nanstd(y)*nanstd(y)/10));
                disp(a/b)
                save_p_value(i,counter) = p;
                counter = counter +1;
            end
        end

    end
end

f=figure('units','normalized','outerposition',[0 0 0.7 0.5]);
%tiledlayout(3,3,'TileSpacing', 'compact')

violinplot(save_p_value',[],'ViolinColor',[0.25 0.25 0.25]	);
yline(0.05,'r')
yline(0.1, 'k')
fontsize(f, 20, "points");

xticks(1:15);
xticklabels(label_signatures_corr_plot);

ylabel('p-value')

filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/significance_test_new.jpg');
exportgraphics(f,filename,'Resolution',300)


%% Summaries of p-values per signature
% Assumes: save_p_value is [15 x nPairs] with NaNs for unused cells
%          label_signatures_corr_plot is 1x15 cell array of signature names

% Median p per signature
median_p = median(save_p_value, 2, 'omitnan');                  % [15 x 1]

% Fraction of p < 0.10 per signature
frac_p_lt_010 = mean(save_p_value < 0.10, 2, 'omitnan');        % [15 x 1], in 0..1

% (Optional) also fraction < 0.05, if you want
frac_p_lt_005 = mean(save_p_value < 0.05, 2, 'omitnan');        % [15 x 1]

% Build overview table (rows = signatures)
T_p_overview = table(median_p, frac_p_lt_010, frac_p_lt_005, ...
    'VariableNames', {'median_p','frac_p_lt_0_10','frac_p_lt_0_05'}, ...
    'RowNames', label_signatures_corr_plot);

disp(T_p_overview);

% Export CSV (keeps row names as first column)
writetable(T_p_overview, '/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/median_frac_p_overview.csv', ...
           'WriteRowNames', true);

%% (Optional) LaTeX table (3 decimals; fraction shown as %)
print_latex = true;
if print_latex
    sigs = T_p_overview.Properties.RowNames;
    mp   = round(T_p_overview.median_p, 3);
    f10  = round(100*T_p_overview.frac_p_lt_0_10, 1);  % percent with 0.1 resolution
    f05  = round(100*T_p_overview.frac_p_lt_0_05, 1);

    fprintf('\\begin{table}[H]\\centering\\caption{Median p-values and fraction of tests with p<0.10 and p<0.05 for each signature.}\\n');
    fprintf('\\renewcommand{\\arraystretch}{1.2}\\begin{tabular}{|l|c|c|c|}\\hline\\n');
    fprintf('Signature & Median $p$ & $\\,\\#(p<0.10)$ [%%] & $\\,\\#(p<0.05)$ [%%] \\\\ \\hline\\n');
    for i = 1:numel(sigs)
        fprintf('%s & %.3f & %.1f & %.1f \\\\ \\hline\\n', sigs{i}, mp(i), f10(i), f05(i));
    end
    fprintf('\\end{tabular}\\label{tab:median_frac_p}\\end{table}\\n');
end
%% CHECK CORRELATION BETWEEN SIGNATURES

corr_results = NaN(15,15,8);

h=figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,2,'TileSpacing', 'compact')

%Correlation_Matrix = NaN(8,9,517);

cmap = brewermap(19,"-RdBu");

for i = 1:length(objective_functions)
    Correlation_Matrix = NaN(10,9);
    nexttile

    % MORE LOOPS

    for j = 1:length(sig_corr_plot)
        Correlation_Matrix(:,j) = reshape(Error_Norm_Values(j,:,i),[],1);

    end
    A = corrcoef(Correlation_Matrix(:,:),"Rows","complete");
    % PLOTTING
    ii = ones(size(A));
    idx = tril(ii);
    A(~idx) = NaN;
    
    hmap = heatmap(A,'Colormap',cmap,'ColorLimits',[-1 1],'MissingDataColor', 'w', ...
        'GridVisible', 'off', 'MissingDataLabel', " ",'XData',label_signatures_corr_plot, ...
        "YData",label_signatures_corr_plot);
    
    title(hmap, OF_Plot{i});  % Customize as needed
    xlabel(hmap, 'Signatures');  % Customize as needed
    ylabel(hmap, 'Signatures');  % Customize as needed
    
    corr_results(:,:,i) = A;
end

filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/correlation_test.jpg');
exportgraphics(h,filename,'Resolution',300)

%% MEAN CORRELATION BETWEEN SIGNATURES (styled like the OBS plot)

corr_results = NaN(15,15,8);   % store each objective's correlation

% ==== build per-objective correlation matrices (PAIRWISE to match OBS) ====
for i = 1:length(objective_functions)
    Correlation_Matrix = NaN(10,9);
    for j = 1:length(sig_corr_plot)
        Correlation_Matrix(:, j) = reshape(Error_Norm_Values(j,:,i), [], 1);
    end
    A = corrcoef(Correlation_Matrix, "Rows", "pairwise");
    corr_results(:,:,i) = A;
end

% ==== mean across objectives ====
Mean_Corr = mean(corr_results, 3, 'omitnan');   % 15x15

% ==== mask upper triangle (keep diagonal/lower) ====
ii  = ones(size(Mean_Corr));
idx = tril(ii);
Mean_Corr(~idx) = NaN;

% ==== single heatmap with same styling as OBS ====
h = figure('units','normalized','outerposition',[0 0 0.7 1]);
cmap = brewermap(19,"-RdBu");

hmap = heatmap( ...
    Mean_Corr, ...
    'Colormap', cmap, ...
    'ColorLimits', [-1 1], ...
    'MissingDataColor', 'w', ...
    'MissingDataLabel', " ", ...
    'GridVisible', 'off', ...
    'XData', label_signatures_corr_plot, ...
    'YData', label_signatures_corr_plot);

title(hmap, 'Mean Correlation of Signatures Across Objective Functions');
xlabel(hmap, 'Signatures');
ylabel(hmap, 'Signatures');
hmap.CellLabelFormat = '%.2f';
hmap.FontSize = 20;

% Show colorbar (if available for your MATLAB version)
try, hmap.ColorbarVisible = 'on'; end

% ==== export ====
filename = '/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/correlation_mean_obsStyle.jpg';
exportgraphics(h, filename, 'Resolution', 300);
%% CHECK CORRELATION BETWEEN SIGNATURES FOR OBSERVATIONS

corr_results_obs = NaN(15,15);

h=figure('units','normalized','outerposition',[0 0 0.7 1]);
%tiledlayout(4,2,'TileSpacing', 'compact')

%Correlation_Matrix = NaN(8,9,517);

cmap = brewermap(19,"-RdBu");

Correlation_Matrix = NaN(10,9);
%nexttile

% MORE LOOPS

for j = 1:length(sig_corr_plot)
    Correlation_Matrix(:,j) = reshape(obs_values(j,:),[],1);
end


A = corrcoef(Correlation_Matrix(:,:),"Rows","pairwise");
% PLOTTING
ii = ones(size(A));
idx = tril(ii);
A(~idx) = NaN;

hmap = heatmap(A,'Colormap',cmap,'ColorLimits',[-1 1],'MissingDataColor', 'w', ...
    'GridVisible', 'off', 'MissingDataLabel', " ",'XData',label_signatures_corr_plot, ...
    "YData",label_signatures_corr_plot);

hmap.CellLabelFormat = '%.2f';
%title(hmap, OF_Plot{i});  % Customize as needed
xlabel(hmap, 'Signatures');  % Customize as needed
ylabel(hmap, 'Signatures');  % Customize as needed
hmap.FontSize = 20;

corr_results(:,:,i) = A;


filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/correlation_test_obs.jpg');
exportgraphics(h,filename,'Resolution',300)

%%

% Get number of signatures (columns)
[n_samples, n_signatures] = size(Correlation_Matrix);

% Prepare a new figure
figure('units','normalized','outerposition',[0 0 1 1]);

for i = 1:n_signatures
    for j = 1:n_signatures

        subplot(n_signatures, n_signatures, (i-1)*n_signatures + j)

        x = Correlation_Matrix(:,j);
        y = Correlation_Matrix(:,i);

        % Remove NaNs
        valid_idx = ~isnan(x) & ~isnan(y);
        x = x(valid_idx);
        y = y(valid_idx);

        % Only plot if enough valid data
        if numel(x) > 1 && numel(y) > 1
            scatter(x, y, 15, 'filled')
        end

        % Diagonal: show variable name
        if i == j
            text(0.5, 0.5, label_signatures_corr_plot{i}, ...
                'Units', 'normalized', 'HorizontalAlignment', 'center', ...
                'FontWeight', 'bold', 'FontSize', 10);
        end

        % Axis formatting
        set(gca, 'XTick', [], 'YTick', []);
        axis tight
    end
end

sgtitle('Pairwise Scatter Plots Between Signatures', 'FontSize', 16)

% Optional: export figure
filename = '/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/pairwise_scatter_obs.jpg';
exportgraphics(gcf, filename, 'Resolution', 300)


%% Combine into one overview figure

coarse_corr = NaN(15,15);

for j = 1:15
    for l = 1:15 
        
%         counter = 0;
%         agree_pos = 0;
%         agree_neg = 0;
%         disagree = 0;
% 
%         for i = 1:8
%             for k = 1:8
%                 if i ~= k && i>k && j>l
%                     if corr_results(j,l,i) > 0 || corr_results(j,l,k) > 0 
%                         agree_pos = agree_pos + 1 ;
%                         
%                     elseif corr_results(j,l,i) < 0 || corr_results(j,l,k) < 0 
%                         agree_neg = agree_neg +1 ;
%                     else
%                         disagree = disagree + 1;
%                     end
%                     counter = counter + 1;
%                 else
% 
%                 end
% 
% 
%             end
%         end
%         if agree_pos >= agree_neg
%             coarse_corr(j,l) = agree_pos/counter;
%         else
%             coarse_corr(j,l) = (-1)*agree_neg/counter;
%         end
        coarse_corr(j,l) = mean(corr_results(j,l,:));
    end
end

h=figure('units','normalized','outerposition',[0 0 0.7 1]);

ii = ones(size(coarse_corr));
idx = tril(ii);
coarse_corr(~idx) = NaN;

hmap = heatmap(coarse_corr,'Colormap',cmap,'ColorLimits',[-1 1],'MissingDataColor', 'w', ...
        'GridVisible', 'off', 'MissingDataLabel', " ",'XData',label_signatures_corr_plot, ...
        "YData",label_signatures_corr_plot);
hmap.CellLabelFormat = '%.2f';
title("Mean Correlation Between Signatures")
xlabel(hmap, 'Signatures');  % Customize as needed
ylabel(hmap, 'Signatures');  % Customize as needed
hmap.FontSize = 20;
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/correlation_test_mean.jpg');
exportgraphics(h,filename,'Resolution',300)

%% NEED TO REORGANIZE ITERATION OF SIGNATURES?

%% CALCULATE IMPACT METRIC V2.0
% This is based on performance by 

Impact_Signatures = NaN(length(sorted_sig_new_significant),517,3);
Error_Values = NaN(8,8,47,11);

for i = 1:length(sorted_sig_new_significant)
    signature = sorted_sig_new_significant{i};

    for j = 1:length(catchments_aridity)
        catchment = catchments_aridity{j};
        for k = 1:length(objective_functions) % number of models (currently: 3)
            %model = model_list_sorted{i};
            obj_fun = objective_functions{k};
            for l = 1:numel(model_list)
                model = model_list{l};
                try 
                    Error_Values(i,k,l,j) = sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun);
                    %if obs_signatures_cali_bench.(signature).(catchment) ~= 0
                        
                        
                        %/obs_signatures_cali_bench.(signature).(catchment);
                    %else
                    %end
                catch
                end
            end
        end
    end
    
    Values_Signature = Error_Values(i,:,:,:);
    
    % CALCULATE IMPACT OFs = k
    % Calculate median over Models and Catchments
    %values_of = NaN(length(objective_functions),1);
    values_of = NaN(517,1);
    index = 0;
    norm_of = (max(Error_Values(i,:,:,:),[],[3,4],'omitnan')-min(Error_Values(i,:,:,:),[],[3,4],'omitnan'));%std(Error_Values(i,:,:,:),0,[3,4],'omitnan');
            
    for j = 1:length(catchments_aridity)
        for l = 1:length(model_list)
            index = index+1;
            disp(index)
            values_of(index,1) = max(Error_Values(i,:,l,j))-min(Error_Values(i,:,l,j));%/median(Error_Values(i,:,l,j),'all','omitnan');
            
            %norm_of = (max(Error_Values(i,:,:,:),[],[3,4],'omitnan')-min(Error_Values(i,:,:,:),[],[3,4],'omitnan'));%std(Error_Values(i,:,:,:),0,[3,4],'omitnan');
            
            %std(values_of,'omitnan')./mean(norm_of,'omitnan');%(max(norm_of,[],'omitnan')-min(norm_of,[],'omitnan'));%/mean(values_of,'omitnan');
        
        end 
    end
    Impact_Signatures(i,:,1)=values_of./median(norm_of,'all','omitnan');
    % CALCULATE IMPACT MODELs = l
    % Calculate median over OFs and Catchments
    values_model = NaN(88,1);
    index = 0;
    norm_model = (max(Error_Values(i,:,:,:),[],[2,4],'omitnan')-min(Error_Values(i,:,:,:),[],[2,4],'omitnan'));
                %norm_of = (max(Error_Values(i,:,:,:),[],[3,4],'omitnan')-min(Error_Values(i,:,:,:),[],[3,4],'omitnan'));%std(Error_Values(i,:,:,:),0,[3,4],'omitnan');

    for j = 1:length(catchments_aridity)
        for k = 1:length(objective_functions)
            index = index+1;
            disp(index)
            values_model(index,1) = max(Error_Values(i,k,:,j))-min(Error_Values(i,k,:,j));%/median(Error_Values(i,k,:,j),'all','omitnan');
            
            %norm_of = (max(Error_Values(i,:,:,:),[],[3,4],'omitnan')-min(Error_Values(i,:,:,:),[],[3,4],'omitnan'));%std(Error_Values(i,:,:,:),0,[3,4],'omitnan');
            %Impact_Signatures(i,1)=std(values_of,'omitnan')./mean(norm_of,'omitnan');%(max(norm_of,[],'omitnan')-min(norm_of,[],'omitnan'));%/mean(values_of,'omitnan');
        
        end
    end
    Impact_Signatures(i,1:88,2)=values_model./median(norm_model,'all','omitnan');

    %values_model = NaN(1,1);
    %values_model = median(Error_Values(i,:,:,:),[2,4],'omitnan');
    %norm_model = (max(Error_Values(i,:,:,:),[],[2,4],'omitnan')-min(Error_Values(i,:,:,:),[],[2,4],'omitnan'));
    %Impact_Signatures(i,2)=std(values_model,'omitnan')./mean(norm_model,'omitnan');%(max(norm_model,[],'omitnan')-min(norm_model,[],'omitnan'));%/mean(values_model,'omitnan');

    % CALCULATE IMPACT CATCHMENTs = j
    % Calculate median over OFs and Models
    values_catch = NaN(376,1);
    index = 0;
    norm_catch = (max(Error_Values(i,:,:,:),[],[2,3],'omitnan')-min(Error_Values(i,:,:,:),[],[2,3],'omitnan'));
   
    for k = 1:length(objective_functions)
        for l = 1:length(model_list)
            index = index+1;
            disp(index)
            values_catch(index,1) = max(Error_Values(i,k,l,:))-min(Error_Values(i,k,l,:));%/median(Error_Values(i,k,l,:),'all','omitnan');
            
            %norm_of = (max(Error_Values(i,:,:,:),[],[3,4],'omitnan')-min(Error_Values(i,:,:,:),[],[3,4],'omitnan'));%std(Error_Values(i,:,:,:),0,[3,4],'omitnan');
            %Impact_Signatures(i,1)=std(values_of,'omitnan')./mean(norm_of,'omitnan');%(max(norm_of,[],'omitnan')-min(norm_of,[],'omitnan'));%/mean(values_of,'omitnan');

        end
    end
    Impact_Signatures(i,1:376,3)=values_catch./median(norm_catch,'all','omitnan');

    %values_catch = NaN(1,1);
    %values_catch = median(Error_Values(i,:,:,:),[2,3],'omitnan');
    %norm_catch = (max(Error_Values(i,:,:,:),[],[2,3],'omitnan')-min(Error_Values(i,:,:,:),[],[2,3],'omitnan'));
    %Impact_Signatures(i,3)=std(values_catch,'omitnan')./mean(norm_catch,'omitnan');%mean(values_catch,'omitnan');

    %fprint('finished signature %f',sorted_signatures{i})
end

%% ANOVA
% Initialize the results matrix to store variability proportions (15x3)
Results_Matrix = zeros(15, 3);

for i = 1:15
    % Extract the subset for the current slice in the first dimension
    subset = squeeze(Error_Values(i, :, :, :));
    
    % Reshape subset into a column vector for anovan
    data = subset(:);
    
    

    % Define factor levels for each dimension
    [dim2, dim3, dim4] = ndgrid(1:size(subset, 1), 1:size(subset, 2), 1:size(subset, 3));
    factors = {dim2(:), dim3(:), dim4(:)};
    
    % Perform ANOVA and get sum of squares (SS) for each factor
    [p, tbl, stats] = anovan(data, factors, 'model', 'interaction', 'display', 'off');
    
    % Extract SS values from the ANOVA table (rows 2, 3, and 4 correspond to factors)
    ss_total = cell2mat(tbl(end, 2));  % Total SS
    ss_dim2 = cell2mat(tbl(2, 2));     % SS for dimension 2
    ss_dim3 = cell2mat(tbl(3, 2));     % SS for dimension 3
    ss_dim4 = cell2mat(tbl(4, 2));     % SS for dimension 4
    
    % Proportion of variability explained by each dimension
    Results_Matrix(i, 1) = ss_dim2 / ss_total;
    Results_Matrix(i, 2) = ss_dim3 / ss_total;
    Results_Matrix(i, 3) = ss_dim4 / ss_total;
end

% Display the Results_Matrix (15x3)
disp(Results_Matrix);


%%

% Initialize the results matrix to store feature importance (15x3)
Results_Matrix_RF = zeros(8, 3);

for i = 1:8
    % Extract the subset for the current slice in the first dimension
    subset = squeeze(Error_Values(i, :, :, :));
    
    % Reshape subset into a column vector for the target variable (y)
    y = subset(:);
    
    % Define the features (factors for each dimension)
    [dim2, dim3, dim4] = ndgrid(1:size(subset, 1), 1:size(subset, 2), 1:size(subset, 3));
    X = [dim2(:), dim3(:), dim4(:)];  % Combine dimensions into a feature matrix
    
    % Train a Random Forest regressor using TreeBagger
    rfModel = TreeBagger(300, X, y, 'Method', 'regression', 'OOBPredictorImportance', 'on');
    
    % Extract feature importance scores
    featureImportance = rfModel.OOBPermutedPredictorDeltaError;
    
    % Normalize importance scores to sum to 1
    Results_Matrix_RF(i, :) = featureImportance / sum(featureImportance);
end

% Display the Results_Matrix_RF (15x3)
disp(Results_Matrix_RF);
%%
% Colors (normalize 0–1)
stack_colors = [
    141,160,203;   % Objective Function
    252,141,98;   % Model
    102,194,165 % Catchment
] ./ 255;
    
ax = figure();

% Stacked bar: each row = signature, 3 columns = components
b = bar(Results_Matrix_RF, 'stacked');
hold on

for k = 1:3
    b(k).FaceColor = stack_colors(k,:);
    b(k).EdgeColor = 'none';
    %b(k).FaceAlpha = 0.6;   % 0 = fully transparent, 1 = fully opaque
end

% Labels & styling (matching your “like here” setup)
legend({'Objective Function','Model','Catchment'}, 'Location','best');
xlim([0.5 8.5])
xticks(1:8)
xticklabels(label_signatures_new_significant);
ylabel('Predictor Importance')
title('Random Forest Feature Importance by Signature')
grid on

% (Optional) since each row is normalized
ylim([0 1])

% Export
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/rf_importance.jpg');
exportgraphics(ax, filename, 'Resolution', 300);

%%
ax = figure();

%Impact_Signatures_median = NaN(15,3);
%Impact_Signatures_sd = NaN(15,3);

for p = 1:8
    for k = 1:3
%        Impact_Signatures_median(p,k) = abs(median(Impact_Signatures(p,:,k),'all','omitnan'));
        %Impact_Signatures_sd = std(Impact_Signatures(p,:,k),'all','omitnan');
    end
end

    %errorbar(1:15,median_median_error(:,k),std_median_error(:,k),'Color',colors(k,:),"LineWidth",2);
    %plot(p,abs(median(Impact_Signatures(p,:,1),std(Impact_Signatures(p,:,1)),'all','omitnan')),'o-','Color','red')
%violinplot(Impact_Signatures(:,:,1)')
%    hold on
%violinplot(Impact_Signatures(:,:,2)')
    %plot(p,abs(median(Impact_Signatures(p,:,1),'all','omitnan')),'-','Color','red')
    %hold on
    %plot(p,abs(median(Impact_Signatures(p,:,2),'all','omitnan')),'-','Color','blue')
    %plot(p,abs(median(Impact_Signatures(p,:,3),'all','omitnan')),'-','Color','black')
%plot(1:15,abs(Impact_Signatures(:,4)),'--','Color','magenta')

colors = {'red','blue','black'};

for k = 1:3
    plot(Results_Matrix_RF(:,k),'-','Color',colors{k}, 'LineWidth', 2)
    hold on
end

legend('Objective Function','Model','Catchment')

xlim([0.5 8.5])
%ylim([0 0.8])

xticks(1:8)
xticklabels(label_signatures_new_significant);
ylabel('Predictor Importance')
grid on

filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/rf_importance.jpg');
exportgraphics(ax,filename,'Resolution',300)

%print(ax, 'impact_signature','-dpng');%, '-dpdf', '-fillpage');
%set(ax, 'XTick', 1:numel(signature_list_plot_sorted), 'XTickLabel', signature_list_plot_sorted);


%% Plot Impact
%Impact_Signatures(:,4)=mean(Impact_Signatures(:,:),2,'omitnan');
ax = figure();

Impact_Signatures_median = NaN(15,3);
Impact_Signatures_sd = NaN(15,3);

for p = 1:15
    for k = 1:3
        Impact_Signatures_median(p,k) = abs(median(Impact_Signatures(p,:,k),'all','omitnan'));
        %Impact_Signatures_sd = std(Impact_Signatures(p,:,k),'all','omitnan');
    end
end

    %errorbar(1:15,median_median_error(:,k),std_median_error(:,k),'Color',colors(k,:),"LineWidth",2);
    %plot(p,abs(median(Impact_Signatures(p,:,1),std(Impact_Signatures(p,:,1)),'all','omitnan')),'o-','Color','red')
%violinplot(Impact_Signatures(:,:,1)')
%    hold on
%violinplot(Impact_Signatures(:,:,2)')
    %plot(p,abs(median(Impact_Signatures(p,:,1),'all','omitnan')),'-','Color','red')
    %hold on
    %plot(p,abs(median(Impact_Signatures(p,:,2),'all','omitnan')),'-','Color','blue')
    %plot(p,abs(median(Impact_Signatures(p,:,3),'all','omitnan')),'-','Color','black')
%plot(1:15,abs(Impact_Signatures(:,4)),'--','Color','magenta')

colors = {'red','blue','black'};

for k = 1:3
    plot(Impact_Signatures_median(:,k),'-','Color',colors{k})
    hold on
end

legend('Objective Function','Model','Catchment')

xlim([0.5 15.5])
%ylim([0 0.8])

xticks(1:15)
xticklabels(signature_list_plot_sorted);
ylabel('Normalized Impact')
grid on

filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/impact_metric.jpg');
exportgraphics(ax,filename,'Resolution',300)

%print(ax, 'impact_signature','-dpng');%, '-dpdf', '-fillpage');
%set(ax, 'XTick', 1:numel(signature_list_plot_sorted), 'XTickLabel', signature_list_plot_sorted);

%% PLOT RESULTS FOR SPECIFIC EFFECTS
% PLOT FOR LOW FLOW COMPARISON

of_low_flow = {'of_log_KGE','of_log_NSE','of_diagnostic_efficiency','of_KGE'};
low_flow_sigs = {'sig_x_percentile_5per','sig_x_Q_duration_low','sig_x_Q_frequency_low'};

low_flow_plot_of = {'log KGE','log NSE','DE','KGE'};
low_flow_plot_sigs = {'Q5','Low Flow Duration','Low Flow Frequency'};

colors =(brewermap(11,"Spectral"));

norm_value = NaN(length(low_flow_sigs),1);

store_plot_norm = NaN(3,47,11,8);
store_intermediate = NaN(3,47,11,8);

for j = 1:length(low_flow_sigs)
    
    %h=figure('units','normalized','outerposition',[0 0 1 1]);
    norm_max_store = zeros(11,1);
    norm_min_store = zeros(11,1);
    norm_max = NaN(11,1);
    norm_min = NaN(11,1);

    for k = 1:length(of_low_flow)
        %h=figure('PaperUnits', 'inches', ...
        %       'PaperSize', [11, 8.5], ...
        %       'PaperPositionMode', 'manual', ...
        %       'PaperPosition', [0, 0, 11, 8.5], ...
        %      'Units', 'inches', ...
        %       'Position', [0, 0, 11, 8.5]); 
        obj_fun = of_low_flow{k};
        store_median_errors = NaN(11,47);
        signature = low_flow_sigs{j};
        %variance_value = NaN(1);

        for l = 1:length(catchments_aridity)
            catchment = catchments_aridity{l};

            for i = 1:length(model_list)
                model = model_list{i};

                try
                    store_intermediate(j,i,l,k) = (sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment));
                catch
                    %disp("Calculation Not Possible")
                end

            end
            %store_median_errors(l,k,j) = median(store_intermediate,"omitnan");
            norm_max(l) = max(store_intermediate(j,:,l,:),[],'all');
            disp(store_intermediate(j,:,l,k))
            norm_min(l) = min(store_intermediate(j,:,l,:),[],'all');

            if norm_max(l) > norm_max_store(l)
                norm_max_store(l) = norm_max(l);
            else

            end

            if norm_min(l) < norm_min_store(l)
                norm_min_store(l) = norm_min(l);
            else

            end

        end
        %norm_value = abs(norm_max-norm_min);
    end
    norm_value(j) = max(abs(store_intermediate(j,:,:,:)),[],'all','omitnan');

end


for j = 1:length(low_flow_sigs)

    h=figure('units','normalized','outerposition',[0 0 0.5 0.3]);
    tiledlayout(1,4,'TileSpacing', 'compact')

    for k = 1:length(of_low_flow)
        
        for u = 1:11
            store_plot_norm(j,:,u,:) = store_intermediate(j,:,u,:)./norm_value(j);
        end

        nexttile
        % Copy the current slice of store_plot_norm for manipulation
        plot_data = squeeze(store_plot_norm(j,:,:,k));

        % Identify columns that are entirely NaN
        nan_columns = all(isnan(plot_data), 1);

        % Replace entirely NaN columns with a placeholder (e.g., 0 or a very small number)
        % This example uses 0 for simplicity. Adjust as needed for your visualization.
        plot_data(:, nan_columns) = -10;
        violinplot(plot_data,1,'ViolinColor',colors(:,:))
        try
            
            % Customize your plotting here to visually indicate or ignore the placeholder values
        catch
            disp('No results')
        end
        %boxplot(store_plot_norm(:,:,t), 'OutlierSize',20,'Symbol','.','whisker', Inf)
        %sgtitle('Total Runoff Ratio for Calibration on KGE-NP')
        sgtitle(h,sprintf('Signature: %s',low_flow_plot_sigs{j}))
        ylabel('Signature Error');

        title(strrep(convertCharsToStrings(low_flow_plot_of{k}),'_','\_'))
        xlim([0.5 11.5])

        xticks(1:11);
        xticklabels(catchments_labels);
    
        ylim([-1 1])

        grid on
        grid minor
        %fontsize(h,25,"points")   
    end

    %for t = 1:length(objective_functions)
    %end
    % Save figure as PDF
    %filename = sprintf('low_flow_%d.pdf', j);
    %saveas(h, filename);
    filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/low_flow_%d.jpg',j);
    exportgraphics(h,filename,'Resolution',300)

    % Close the figure to avoid displaying each one
    %close(h);
end

%% 4x2 Plots for each signature

%colors =(brewermap(10,"Spectral"));

colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];


yrange_values = [0.3,0.3,1,6,100,100,0.4,0.5,50,5,2,0.6,1,1.2,0.7];
signature_list_plot_sorted = {'Total RR','Event RR','Q5 (mm/d)','Q95 (mm/d)',...
'High Flow Duration (d)','Low Flow Duration (d)','High Flow Frequency',...
'Low Flow Frequency','Mean Half Flow Date (DOY)','FDC Slope','Variability Index','Baseflow Index',...
'BF Recession Coefficient (1/d)','Flashiness Index','Rising Limb Density (1/d)'};

for j = 1:length(sorted_signatures)
    h=figure('units','normalized','outerposition',[0 0 0.5 0.6]); 

    tiledlayout(2,4,'TileSpacing', 'compact')

    %h=figure('units','normalized','outerposition',[0 0 1 1]);
    
    store_plot_norm = NaN(47,10,8);
    store_intermediate = NaN(47,10,8);

    for k = 1:length(objective_functions)
        obj_fun = objective_functions{k};
        store_median_errors = NaN(10,47);
        signature = sorted_signatures{j};
        %variance_value = NaN(1);

        for l = 1:length(catchments_aridity)
            catchment = catchments_aridity{l};

            for i = 1:length(model_list)
                model = model_list{i};

                try
                    store_intermediate(i,l,k) = (sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment));
                catch
                    %disp("Calculation Not Possible")
                end

            end
            %store_median_errors(l,k,j) = median(store_intermediate,"omitnan");


        end
               
        for u = 1:10
            %store_plot_norm(:,u,:) = store_intermediate(:,u,:);%./norm_value(u);
        end

        nexttile
        % Copy the current slice of store_plot_norm for manipulation
        plot_data = store_intermediate(:,:,k);

        % Identify columns that are entirely NaN
        nan_columns = all(isnan(plot_data), 1);

        % Replace entirely NaN columns with a placeholder (e.g., 0 or a very small number)
        % This example uses 0 for simplicity. Adjust as needed for your visualization.
        plot_data(:, nan_columns) = -10;
        %boxplot(plot_data,"Whisker",inf)
        vp = violinplot(plot_data,1,'ViolinColor',colors(k,:));
        
        for v = 1:length(vp)
            vp(v).ScatterPlot.SizeData = 10;   % default is usually ~36
        end

        try
            
            % Customize your plotting here to visually indicate or ignore the placeholder values
        catch
            disp('No results')
        end
        %boxplot(store_plot_norm(:,:,t), 'OutlierSize',20,'Symbol','.','whisker', Inf)
        %sgtitle('Total Runoff Ratio for Calibration on KGE-NP')
        sgtitle(h,sprintf('Signature: %s',signature_list_plot_sorted{j}))
        ylabel('Signature Error');
        
        title(strrep(convertCharsToStrings(OF_Plot{k}),'_','\_'))
        xlim([0.5 10.5])

        xticks(1:10);
        xticklabels(catchments_labels);
    
        ylim([-yrange_values(j) yrange_values(j)])

        grid on
        grid minor
        %fontsize(h,25,"points")   
    end

    %for t = 1:length(objective_functions)
    %end
    % Save figure as PDF
    filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/signature_model_%d.jpg',j );
    exportgraphics(h,filename,'Resolution',300)

    %filename = sprintf('best_of_%d.pdf', j);
    %saveas(h, filename);
    
    % Close the figure to avoid displaying each one
    %close(h);
end

%% Details for best objective functions

of_best_of = {'of_KGE','of_NSE','of_KGE_non_parametric','of_SHE','of_diagnostic_efficiency'};
best_of_sigs = {'sig_TotalRR','sig_x_percentile_95per','sig_BFI'};

best_of_plot_of = {'KGE','NSE','KGE-NP','SHE','DE'};
best_of_plot_sigs = {'Total Runoff Ratio','95th Flow Percentile','Baseflow Index'};

colors =(brewermap(11,"Spectral"));

for j = 1:length(best_of_sigs)
    h=figure('units','normalized','outerposition',[0 0 0.5 0.3]); 

    tiledlayout(1,5,'TileSpacing', 'compact')

    %h=figure('units','normalized','outerposition',[0 0 1 1]);
    
    norm_max_store = zeros(11,1);
    norm_min_store = zeros(11,1);
    norm_max = NaN(11,1);
    norm_min = NaN(11,1);
    store_plot_norm = NaN(47,11,8);
    store_intermediate = NaN(47,11,8);

    for k = 1:length(of_best_of)
        %h=figure('PaperUnits', 'inches', ...
        %       'PaperSize', [11, 8.5], ...
        %       'PaperPositionMode', 'manual', ...
        %       'PaperPosition', [0, 0, 11, 8.5], ...
        %      'Units', 'inches', ...
        %       'Position', [0, 0, 11, 8.5]); 
        obj_fun = of_best_of{k};
        store_median_errors = NaN(11,47);
        signature = best_of_sigs{j};
        %variance_value = NaN(1);

        for l = 1:length(catchments_aridity)
            catchment = catchments_aridity{l};

            for i = 1:length(model_list)
                model = model_list{i};

                try
                    store_intermediate(i,l,k) = (sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun)-obs_signatures_cali_bench.(signature).(catchment));
                catch
                    %disp("Calculation Not Possible")
                end

            end
            %store_median_errors(l,k,j) = median(store_intermediate,"omitnan");
            norm_max(l) = max(store_intermediate(:,l,:),[],'all');
            disp(store_intermediate(:,l,k))
            norm_min(l) = min(store_intermediate(:,l,:),[],'all');

            if norm_max(l) > norm_max_store(l)
                norm_max_store(l) = norm_max(l);
            else
            end

            if norm_min(l) < norm_min_store(l)
                norm_min_store(l) = norm_min(l);
            else

            end

        end
        norm_value = abs(norm_max-norm_min);
        
        for u = 1:11
            store_plot_norm(:,u,:) = store_intermediate(:,u,:);%./norm_value(u);
        end

        nexttile
        % Copy the current slice of store_plot_norm for manipulation
        plot_data = store_plot_norm(:,:,k);

        % Identify columns that are entirely NaN
        nan_columns = all(isnan(plot_data), 1);

        % Replace entirely NaN columns with a placeholder (e.g., 0 or a very small number)
        % This example uses 0 for simplicity. Adjust as needed for your visualization.
        plot_data(:, nan_columns) = -10;
        %boxplot(plot_data,"Whisker",inf)
        violinplot(plot_data,1,'ViolinColor',colors(:,:))
        try
            
            % Customize your plotting here to visually indicate or ignore the placeholder values
        catch
            disp('No results')
        end
        %boxplot(store_plot_norm(:,:,t), 'OutlierSize',20,'Symbol','.','whisker', Inf)
        %sgtitle('Total Runoff Ratio for Calibration on KGE-NP')
        sgtitle(h,sprintf('Signature: %s',best_of_plot_sigs{j}))
        ylabel('Signature Error');
        
        title(strrep(convertCharsToStrings(best_of_plot_of{k}),'_','\_'))
        xlim([0.5 11.5])

        xticks(1:11);
        xticklabels(catchments_labels);
    
        ylim([-0.4 0.4])

        grid on
        grid minor
        %fontsize(h,25,"points")   
    end

    %for t = 1:length(objective_functions)
    %end
    % Save figure as PDF
    filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/best_of_%d.jpg',j );
    exportgraphics(h,filename,'Resolution',300)
    %filename = sprintf('best_of_%d.pdf', j);
    %saveas(h, filename);
    
    % Close the figure to avoid displaying each one
    %close(h);
end

%% FUNCTIONS
function customColormap = greenCenteredColormap(n)
    % Default number of colors
    if nargin < 1
        n = 256; % Default number of colors in the colormap
    end

    % Create the gradient
    half = floor(n / 2);
    
    % From blue to green
    blueToGreen = [linspace(0, 0, half)', linspace(0, 1, half)', linspace(1, 0, half)'];
    
    % From green to red
    greenToRed = [linspace(0, 1, n - half)', linspace(1, 0, n - half)', linspace(0, 0, n - half)'];
    
    % Combine the two gradients
    customColormap = [blueToGreen; greenToRed];
    
    % Handle the case where n is odd
    if mod(n, 2) == 1
        customColormap = [customColormap; [0 1 0]];
    end
end
