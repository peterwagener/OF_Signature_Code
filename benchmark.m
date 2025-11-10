%% Create Benchmark models

clear

addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/TOSSH-master/'))
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/marrmot_211/'))
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/ma_thesis/'))
%addpath(genpath(''))

% load data
%load('signatures_2050.mat')
load('additional.mat')
%load('forcing.mat')
%load('Q_OF.mat')

%objective_functions = obj_funs;
%current_signatures = signatures;
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155','camels_12381400','camels_02017500',...
    'camelsgb_39037','camels_03460000','hysets_01AF007','camelsgb_27035','camelsgb_8013','lamah_200048'};

signatures_Q = {'sig_FDC_slope','sig_RisingLimbDensity',...
    'sig_BaseflowRecessionK','sig_HFD_mean',...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex'};
signatures_Q_P = {'sig_EventRR','sig_TotalRR'};
signatures_high_low = {'sig_x_Q_duration','sig_x_Q_frequency'};
signatures_percentage = {'sig_x_percentile'};

current_signatures = [signatures_Q,signatures_Q_P,signatures_high_low,signatures_percentage];

path_nc = '/Users/peterwagener/Desktop/ma_thesis_dump/ma_thesis/catchments_new/';
Files=dir(fullfile(path_nc,'*.nc'));
% catchments = cell(length(Files)+1,1);
% for k = 1:length(Files)
%     FileNames=Files(k).name;
%     catchments{k} = FileNames(1:end-3);
% end

% select catchments

%% Read Data
% read data
% catchments from Caravan (NetCDF)
% ncdisp('/Users/peterwagener/Desktop/ma_thesis/lamah_212670.nc')
%lon = ncread('/Users/peterwagener/Desktop/ma_thesis/lamah_212670.nc','longitude')

precip = zeros(14609,length(catchments_aridity));
temp = zeros(14609,length(catchments_aridity));
pet = zeros(14609,length(catchments_aridity));
streamflow = zeros(14609,length(catchments_aridity));

% write struct/cell with data (precip, temp, pet, streamflow)
for h = 1:11
    precip(:,h)   = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'total_precipitation_sum');
    temp(:,h)     = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'temperature_2m_mean');
    pet(:,h)      = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'potential_evaporation_sum');
    streamflow(:,h) = ncread(fullfile(path_nc,strcat(catchments_aridity{h},'.nc')),'streamflow');
end

%% 1. Build date_array
start_date = datetime(1981,1,2);
end_date   = datetime(2020,12,31);
date_array = (start_date:end_date)';

%% 2. Ensure consistent length
T = min([numel(date_array), size(precip,1), size(streamflow,1), size(temp,1)]);
precip     = precip(1:T,:);
streamflow = streamflow(1:T,:);
temp       = temp(1:T,:);
date_array = date_array(1:T);

%% 3. Convert temp if needed
if mean(temp(:),'omitnan') > 200
    temp = temp - 273.15; % Kelvin → °C
end

%% 4. Annual means in mm/yr
yrs = year(date_array);
[uY,~,yIdx] = unique(yrs);

sum_by_year  = @(x) accumarray(yIdx,x,[numel(uY),1],@(v)sum(v,'omitnan'),NaN);
mean_by_year = @(x) accumarray(yIdx,x,[numel(uY),1],@(v)mean(v,'omitnan'),NaN);

M = size(precip,2);
LT_mean_annual_precip = nan(1,M);
LT_mean_annual_stream = nan(1,M);
LT_mean_temp          = nan(1,M);

for j = 1:M
    annP = sum_by_year(precip(:,j));
    annQ = sum_by_year(streamflow(:,j));
    annT = mean_by_year(temp(:,j));

    LT_mean_annual_precip(j) = mean(annP,'omitnan');
    LT_mean_annual_stream(j) = mean(annQ,'omitnan');
    LT_mean_temp(j)          = mean(annT,'omitnan');
end

%% 5. Print results
fprintf('%-12s  %14s  %14s  %12s\n','Catchment','Precip [mm/yr]','Streamflow [mm/yr]','Temp [°C]');
for j = 1:M
    fprintf('%-12s  %14.1f  %14.1f  %12.2f\n',catchments_aridity{j}, ...
        LT_mean_annual_precip(j),LT_mean_annual_stream(j),LT_mean_temp(j));
end
%% Calculate OF value for each model, catchment and OF
% Get runoff, remove negative values and store runoff

numCatchments = numel(catchments);
numFunctions = numel(objective_functions);
numModels = numel(model_list);

baseDirectory = '/Users/peterwagener/Desktop/ma_thesis_dump/output3';

% Create a cell array to store the results
Q_sim_cali = cell(numCatchments, numFunctions, numModels);
Q_sim_vali = cell(numCatchments, numFunctions, numModels);
    
% Generate datetime array from 1981-01-02 to 2020-12-31
start_year = 1981;
end_year = 2020;
date_array = datetime(start_year, 1, 2):datetime(end_year, 12, 31);

for catchmentIdx = 1:numCatchments
    catchment = catchments_aridity{catchmentIdx};
    if catchmentIdx == 3 %% CHANGE TO 3 WHEN NEW DATA AVAILABLE
        start_date = datetime(1990, 1, 1);
        end_date = datetime(1999, 12, 31);
        mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
        cal_idx = find(mask_cali_idx);

        start_date = datetime(1982, 1, 1);
        end_date = datetime(1988, 12, 31);
        mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
        eval_idx = find(mask_eval_idx);

        %cal_idx = 2924+365:2924+364+365*10;
        %eval_idx = 1+365:1+364+365*7;
    else
        start_date = datetime(2005, 1,1);
        end_date = datetime(2014,12,31);
        mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
        cal_idx = find(mask_cali_idx);

        start_date = datetime(1994, 1, 1);
        end_date = datetime(2003, 12, 31);
        mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
        eval_idx = find(mask_eval_idx);
    end
    
    for funcIdx = 1:numFunctions
        obj_fun = objective_functions{funcIdx};
        
        for modelIdx = 1:numModels
            model = model_list{modelIdx};
            directory = fullfile(baseDirectory,catchment, obj_fun, model);
            
            % Load the 'results.mat' file from the directory
            resultsFile = fullfile(directory, 'results.mat');
            
            % Check if the directory path is empty
            try
                if exist(resultsFile,'file')
                    loadedData = load(resultsFile);
                else
                    resultsFile = fullfile(directory, 'results.mat');
                    loadedData = load(resultsFile);
                end

                runoff_sim_eval = loadedData.runoff_sim_eval{1,1};
                if catchmentIdx == 3
                    runoff_sim = loadedData.runoff_sim{1,1};
                else
                    runoff_sim = loadedData.runoff_sim{1,1}(2:3653);
                end
                    streamflow_single = loadedData.streamflow;
                %par_opt = loadedData.optimal_parameters{1,1};

                % REMOVE NEGATIVE VALUES
                runoff_sim_eval(runoff_sim_eval < 0) = 0;
                runoff_sim(runoff_sim < 0) = 0;
                streamflow_single(streamflow_single < 0) = 0;

                % SAVE RUNOFF
                Q_sim_cali{catchmentIdx, funcIdx, modelIdx} = runoff_sim;
                Q_sim_vali{catchmentIdx, funcIdx, modelIdx} = runoff_sim_eval;
                try
                    if obj_fun == "of_SHE"
                        OF_value_cali.(catchment).(obj_fun).(model) = feval(obj_fun,streamflow_single(cal_idx),Q_sim_cali{catchmentIdx, funcIdx, modelIdx},precip(cal_idx,catchmentIdx));
                        OF_value_vali.(catchment).(obj_fun).(model) = feval(obj_fun,streamflow_single(eval_idx),Q_sim_vali{catchmentIdx, funcIdx, modelIdx},precip(eval_idx,catchmentIdx));
                    else
                        OF_value_cali.(catchment).(obj_fun).(model) = feval(obj_fun,streamflow_single(cal_idx),Q_sim_cali{catchmentIdx, funcIdx, modelIdx});
                        OF_value_vali.(catchment).(obj_fun).(model) = feval(obj_fun,streamflow_single(eval_idx),Q_sim_vali{catchmentIdx, funcIdx, modelIdx});
                    end
                catch
                    fprintf('OF could not be calculated \n')
                end
                
                
            catch
                fprintf('file does not exist \n')
                
            end 
        end
    end
end

%% Calculate Benchmark based on interannual flow
start_year = 1981;
end_year = 2020;

date_array = datetime(start_year, 1, 2):datetime(end_year, 12, 31);

% Sample data matrix with NaN values: Replace this with your actual data
num_rows = numel(date_array);
num_cols = 11;

% Extract the data for the years 1994 to 2003
start_date = datetime(2005, 1, 1);
end_date = datetime(2014, 12, 31);
mask = (date_array >= start_date) & (date_array <= end_date);
%mask_new = find(mask);
date_array = date_array(mask);
streamflow_new = streamflow(mask, :);
precip_new = precip(mask,:);

% Calculate the day of the year for each date
days_of_year = day(date_array, 'dayofyear');

% Initialize arrays to store cumulative sum and count for leap years and normal years
cumulative_sum_leap = zeros(366, num_cols); % Account for a possible leap year
num_years_available_leap = zeros(366, num_cols); % Account for a possible leap year

cumulative_sum_normal = zeros(365, num_cols); % Normal year (non-leap year)
num_years_available_normal = zeros(365, num_cols); % Normal year (non-leap year)

% Loop through each day of the year and accumulate the streamflow data
for day_idx = 1:numel(date_array)
    day_data = streamflow_new(day_idx, :);
    day_current = days_of_year(day_idx);
    if isleap(year(date_array(day_idx)))
        day_data_leap = day_data(~isnan(day_data));
        cumulative_sum_leap(day_current, ~isnan(day_data)) = cumulative_sum_leap(day_current, ~isnan(day_data)) + day_data_leap;
        num_years_available_leap(day_current,:) = num_years_available_leap(day_current,:) + ~isnan(day_data);
    else
        day_data_normal = day_data(~isnan(day_data));
        cumulative_sum_normal(day_current, ~isnan(day_data)) = cumulative_sum_normal(day_current, ~isnan(day_data)) + day_data_normal;
        num_years_available_normal(day_current,:) = num_years_available_normal(day_current,:) + ~isnan(day_data);
    end
end

% Calculate the interannual mean for each day of the year, considering leap years and normal years
interannual_mean_leap = cumulative_sum_leap ./ num_years_available_leap;
interannual_mean_normal = cumulative_sum_normal ./ num_years_available_normal;

% set NaN values to 0
nan_values_leap = isnan(interannual_mean_leap);
interannual_mean_leap(nan_values_leap) = 0;

nan_values_normal = isnan(interannual_mean_normal);
interannual_mean_normal(nan_values_normal) = 0;

% Display the interannual means for each day of the year (columns represent different months or data attributes)
%disp("Interannual Mean for Leap Years:");
%disp(interannual_mean_leap);

%disp("Interannual Mean for Normal Years:");
%disp(interannual_mean_normal);

clear day

%% Update values for catchment camelsaus_607155
start_year=1981;
end_year=2020;
date_array = datetime(start_year, 1, 2):datetime(end_year, 12, 31);

start_date = datetime(1990, 1, 1);
end_date = datetime(1999, 12, 31);
mask = (date_array >= start_date) & (date_array <= end_date);
date_array = date_array(mask);
streamflow_new(:,3) = streamflow(mask, 3);
precip_new(:,3)=precip(mask,3);

% Calculate the day of the year for each date
days_of_year = day(date_array, 'dayofyear');

% Initialize arrays to store cumulative sum and count for leap years and normal years
cumulative_sum_leap = zeros(366, num_cols); % Account for a possible leap year
num_years_available_leap = zeros(366, num_cols); % Account for a possible leap year

cumulative_sum_normal = zeros(365, num_cols); % Normal year (non-leap year)
num_years_available_normal = zeros(365, num_cols); % Normal year (non-leap year)

% Loop through each day of the year and accumulate the streamflow data
for day_idx = 1:numel(date_array)
    day_data = streamflow_new(day_idx, 3);
    day_current = days_of_year(day_idx);
    if isleap(year(date_array(day_idx)))
        day_data_leap = day_data(~isnan(day_data));
        cumulative_sum_leap(day_current, ~isnan(day_data)) = cumulative_sum_leap(day_current, ~isnan(day_data)) + day_data_leap;
        num_years_available_leap(day_current,:) = num_years_available_leap(day_current,:) + ~isnan(day_data);
    else
        day_data_normal = day_data(~isnan(day_data));
        cumulative_sum_normal(day_current, ~isnan(day_data)) = cumulative_sum_normal(day_current, ~isnan(day_data)) + day_data_normal;
        num_years_available_normal(day_current,:) = num_years_available_normal(day_current,:) + ~isnan(day_data);
    end
end

% Calculate the interannual mean for each day of the year, considering leap years and normal years
interannual_mean_leap(:,3) = cumulative_sum_leap(:,1) ./ num_years_available_leap(:,1);
interannual_mean_normal(:,3) = cumulative_sum_normal(:,1) ./ num_years_available_normal(:,1);

% set NaN values to 0
nan_values_leap = isnan(interannual_mean_leap);
interannual_mean_leap(nan_values_leap) = 0;

nan_values_normal = isnan(interannual_mean_normal);
interannual_mean_normal(nan_values_normal) = 0;

%% Get 10-year time series and calculate threshold
interannual_mean_store = zeros(365,11);
% create time series
num_catchments = 11;
time_series_10years = NaN(3652, num_catchments); % 10 years * 365 days = 3652 data points
start_year = 2005;

leap_counter = 0;
% Fill in the time series with the interannual means

for year_idx = 1:10
    is_leap_year = isleap(start_year + year_idx - 1);
    days_in_year = 365; % Leap year has 366 days
    interannual_mean = interannual_mean_normal;
    if is_leap_year
        days_in_year = 366; % Normal year has 365 days
        interannual_mean = interannual_mean_leap;
    end
        
    for day_idx = 1:days_in_year
        day_of_year = (year_idx - 1) * 365 + day_idx+leap_counter;
        time_series_10years(day_of_year, :) = interannual_mean(day_idx,:);
    end
    leap_counter = leap_counter + is_leap_year;
end


% Display the 10-year time series for each catchment (columns represent different catchments)
disp("10-Year Time Series for Each Catchment:");
disp(time_series_10years);

% calculate threshold for each 
threshold = zeros(4,11);
for c = 1:11
    for of = 1:8
        obj_fun = objective_functions{of};
        if obj_fun == "of_SHE"
            threshold(of,c) = feval(obj_fun,streamflow_new(:,c),time_series_10years(:,c),precip_new(:,c));
        else
            threshold(of,c) = feval(obj_fun,streamflow_new(:,c),time_series_10years(:,c));
        end
        if threshold(1,c) < 0
            threshold(1,c) = 0;
        elseif threshold(2,c) < -0.4142
            threshold(2,c) = -0.4142;
        else
        end
    end
    figure('Name',['Catchment: ',catchments_aridity{c}]);
    for i = 1:10
        plot(streamflow_new((i-1)*365+1:i*365,c), 'Color','blue')
        hold on
    end
    plot(interannual_mean_normal(:,c),'Color','r')
    hold off
end

%% Select catchments based on benchmark
catchments_labels = {'AUS1','BR6','AUS6','C12','C02','GB3','C02','HYS','GB2','GB8','LAM'};
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

model_counter = zeros(8,11);

for catchmentIdx = 1:11
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


%% Calculate signatures again for benchmark models

date_array = datetime(1981, 1, 2):datetime(2020, 12, 31);

for catchmentIdx = 1:11
    catchment = catchments_aridity{catchmentIdx};

    if catchmentIdx == 3 %'camelsaus_607155' 
        start_date = datetime(1990, 1, 1);
        end_date = datetime(1999, 12, 31);
        mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
        cal_idx = find(mask_cali_idx);

        start_date = datetime(1982, 1, 1);
        end_date = datetime(1988, 12, 31);
        mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
        eval_idx = find(mask_eval_idx);

        %cal_idx = 2924+365:2924+364+365*10;
        %eval_idx = 1+365:1+364+365*7;
    else
        start_date = datetime(2005, 1,1);
        end_date = datetime(2014,12,31);
        mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
        cal_idx = find(mask_cali_idx);

        start_date = datetime(1994, 1, 1);
        end_date = datetime(2003, 12, 31);
        mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
        eval_idx = find(mask_eval_idx);

        %cal_idx = 8401+365:8401+364+3650;
        %eval_idx = 4384+365:4384+364+3650;    
    end

    for objectiveIdx = 1:8
        obj_fun = objective_functions{objectiveIdx};
        for modelIdx = 1:47
            model = model_list{modelIdx};
            try
                if isfield(OF_value_cali_benchmark.(catchment).(obj_fun),(model))
                    directory = fullfile(baseDirectory,catchment, obj_fun, model);
    
                    % Load the 'results.mat' file from the directory
                    resultsFile = fullfile(directory, 'results.mat');

                    % Check if the directory path is empty

                    if exist(resultsFile,'file')
                        loadedData = load(resultsFile);
                    else
                        fprintf('Empty directory path for catchment %s, function %s, and model %s\n', catchment, obj_fun, model);
                        resultsFile = fullfile(directory, 'results.mat');
                        loadedData = load(resultsFile);

                    end

                        
                    for p = 1:length(current_signatures) % currently: 2
        
                        signature = current_signatures{p};
        
                    % CALIBRATION PERIOD
        
                        if any(strcmp(signatures_Q,signature))
                            Signatures_Q_struct_bench.(signature).(catchment).(model).(obj_fun) = feval(signature,Q_sim_cali{catchmentIdx, objectiveIdx, modelIdx},cal_idx);
                            obs_signatures_Q_bench.(signature).(catchment) = feval(signature,streamflow(cal_idx,catchmentIdx),cal_idx);
        
                        elseif any(strcmp(signatures_Q_P,signature))
                            Signatures_Q_P_struct_bench.(signature).(catchment).(model).(obj_fun)= feval(signature,Q_sim_cali{catchmentIdx, objectiveIdx, modelIdx},cal_idx,precip(cal_idx,catchmentIdx));  
                            obs_signatures_Q_P_bench.(signature).(catchment) = feval(signature,streamflow(cal_idx,catchmentIdx),cal_idx,precip(cal_idx,catchmentIdx));
        
                        elseif any(strcmp(signatures_high_low,signature))
                            % calculate high
                            dummy = strcat(signature,convertStringsToChars('_high'));
                            Signatures_high_low_struct_bench.(dummy).(catchment).(model).(obj_fun)= feval(signature,Q_sim_cali{catchmentIdx, objectiveIdx, modelIdx},cal_idx,'high');  
                            obs_signatures_high_low_bench.(dummy).(catchment) = feval(signature,streamflow(cal_idx,catchmentIdx),cal_idx,'high');
                            % calculate low
                            dummy2 = strcat(signature,convertStringsToChars('_low'));
                            Signatures_high_low_struct_bench.(dummy2).(catchment).(model).(obj_fun)= feval(signature,Q_sim_cali{catchmentIdx, objectiveIdx, modelIdx},cal_idx,'low');  
                            obs_signatures_high_low_bench.(dummy2).(catchment) = feval(signature,streamflow(cal_idx,catchmentIdx),cal_idx,'low');
                            
                        else
                            % calculate 5%
                            dummy = strcat(signature,convertStringsToChars('_5per'));
                            Signatures_perc_struct_bench.(dummy).(catchment).(model).(obj_fun)= feval(signature,Q_sim_cali{catchmentIdx, objectiveIdx, modelIdx},cal_idx,5);  
                            obs_signatures_perc_bench.(dummy).(catchment) = feval(signature,streamflow(cal_idx,catchmentIdx),cal_idx,5);
                            % calculate 95% 
                            dummy2 = strcat(signature,convertStringsToChars('_95per'));
                            Signatures_perc_struct_bench.(dummy2).(catchment).(model).(obj_fun)= feval(signature,Q_sim_cali{catchmentIdx, objectiveIdx, modelIdx},cal_idx,95);  
                            obs_signatures_perc_bench.(dummy2).(catchment) = feval(signature,streamflow(cal_idx,catchmentIdx),cal_idx,95);
        
                        end
                            
                   % EVALUATION PERIOD
        
                        if any(strcmp(signatures_Q,signature))
                            Signatures_Q_struct_eval_bench.(signature).(catchment).(model).(obj_fun) = feval(signature,Q_sim_vali{catchmentIdx, objectiveIdx, modelIdx},eval_idx);
                            obs_signatures_Q_eval_bench.(signature).(catchment) = feval(signature,streamflow(eval_idx,catchmentIdx),eval_idx);
        
                        elseif any(strcmp(signatures_Q_P,signature))
                            Signatures_Q_P_struct_eval_bench.(signature).(catchment).(model).(obj_fun)= feval(signature,Q_sim_vali{catchmentIdx, objectiveIdx, modelIdx},eval_idx,precip(eval_idx,catchmentIdx));  
                            obs_signatures_Q_P_eval_bench.(signature).(catchment) = feval(signature,streamflow(eval_idx,catchmentIdx),eval_idx,precip(eval_idx,catchmentIdx));
        
                        elseif any(strcmp(signatures_high_low,signature))
                            % calculate high
                            dummy = strcat(signature,convertStringsToChars('_high'));
                            Signatures_high_low_struct_eval_bench.(dummy).(catchment).(model).(obj_fun)= feval(signature,Q_sim_vali{catchmentIdx, objectiveIdx, modelIdx},eval_idx,'high');  
                            obs_signatures_high_low_eval_bench.(dummy).(catchment) = feval(signature,streamflow(eval_idx,catchmentIdx),eval_idx,'high');
                            % calculate low
                            dummy2 = strcat(signature,convertStringsToChars('_low'));
                            Signatures_high_low_struct_eval_bench.(dummy2).(catchment).(model).(obj_fun)= feval(signature,Q_sim_vali{catchmentIdx, objectiveIdx, modelIdx},eval_idx,'low');  
                            obs_signatures_high_low_eval_bench.(dummy2).(catchment) = feval(signature,streamflow(eval_idx,catchmentIdx),eval_idx,'low');
                            
                        else
                            % calculate 5%
                            dummy = strcat(signature,convertStringsToChars('_5per'));
                            Signatures_perc_struct_eval_bench.(dummy).(catchment).(model).(obj_fun)= feval(signature,Q_sim_vali{catchmentIdx, objectiveIdx, modelIdx},eval_idx,5);  
                            obs_signatures_perc_eval_bench.(dummy).(catchment) = feval(signature,streamflow(eval_idx,catchmentIdx),eval_idx,5);
                            % calculate 95% 
                            dummy2 = strcat(signature,convertStringsToChars('_95per'));
                            Signatures_perc_struct_eval_bench.(dummy2).(catchment).(model).(obj_fun)= feval(signature,Q_sim_vali{catchmentIdx, objectiveIdx, modelIdx},eval_idx,95);  
                            obs_signatures_perc_eval_bench.(dummy2).(catchment) = feval(signature,streamflow(eval_idx,catchmentIdx),eval_idx,95);
        
                        end
    
                    end

    
                else
                    fprintf('model %s did not pass the benchmark for catchment %s and OF %s \n',model,catchment,obj_fun);
                end
            catch
            end
        end
    end
end

sim_signatures_cali_bench = catstruct(Signatures_Q_struct_bench,Signatures_Q_P_struct_bench, Signatures_high_low_struct_bench,Signatures_perc_struct_bench);
obs_signatures_cali_bench = catstruct(obs_signatures_Q_bench,obs_signatures_Q_P_bench,obs_signatures_high_low_bench,obs_signatures_perc_bench);
sim_signatures_eval_bench = catstruct(Signatures_Q_struct_eval_bench,Signatures_Q_P_struct_eval_bench,...
    Signatures_high_low_struct_eval_bench,Signatures_perc_struct_eval_bench);
obs_signatures_eval_bench = catstruct(obs_signatures_Q_eval_bench,obs_signatures_Q_P_eval_bench,...
    obs_signatures_high_low_eval_bench,obs_signatures_perc_eval_bench);

save signatures.mat sim_signatures_cali_bench obs_signatures_cali_bench sim_signatures_eval_bench obs_signatures_eval_bench threshold
save OF_values.mat OF_value_vali_benchmark OF_value_cali_benchmark OF_value_cali OF_value_vali Q_sim_cali Q_sim_vali

%% streamflow comparison
% Feshie = easy to beat benchmark
% Define date range
start_date = datenum('01-Jan-1995');
end_date = datenum('31-Dec-1995');
date_range = start_date:10:end_date;
date_labels = datestr(date_range, 'dd-mmm-yyyy');

% Create the first figure
figure('units','normalized','outerposition',[0 0 1 1]);

% Subplot 1: Observed vs. Benchmark for Feshie (easy to beat benchmark)
subplot(2, 1, 1);
observed_data = streamflow(4384+730:4384+729+365, 10);
observed_dates = start_date + (1:365); % Generate corresponding dates for the observed data
plot(observed_dates, observed_data);
hold on;
plot(observed_dates, interannual_mean_normal(1:end, 10)); % Adjust interannual_mean_normal accordingly
best_model_data = Q_sim_vali{10, 1, 31}(366:730);  % Adjust indexing for the best model data
%best_model_data = Q_sim_vali{11, 1, 7}(366:730);
plot(observed_dates, best_model_data);
legend({'observed', 'benchmark', 'best model'});
title('Feshie - Streamflow Comparison');
xlabel('Date');
ylabel('Streamflow [mm/d]')
xticks(date_range);
xticklabels(date_labels);
xtickangle(45);
datetick('x', 'dd-mm-yy', 'keepticks');

% Subplot 2: Observed vs. Benchmark for Lamah (hard to beat benchmark)
subplot(2, 1, 2);
observed_data = streamflow(4384+730:4384+729+365, 11);
observed_dates = start_date + (1:365); % Generate corresponding dates for the observed data
plot(observed_dates, observed_data);
hold on;
plot(observed_dates, interannual_mean_normal(1:end, 11)); % Adjust interannual_mean_normal accordingly
best_model_data = Q_sim_vali{11, 1, 34}(366:730);  % Adjust indexing for the best model data
%best_model_data = Q_sim_vali{11, 1, 7}(366:730);
plot(observed_dates, best_model_data);
legend({'observed', 'benchmark', 'best model'});
title('Lamah - Streamflow Comparison');
xlabel('Date');
ylabel('Streamflow [mm/d]')
xticks(date_range);
xticklabels(date_labels);
xtickangle(45);
datetick('x', 'dd-mm-yy', 'keepticks');

f = gcf;

fontsize(f,16,"points")
filename = sprintf('/Users/peterwagener/Desktop/ma_thesis_dump/graphics_new/benchmark_sf.jpg');
exportgraphics(f,filename,'Resolution',300)

% ... Add other subplots or comparisons if needed