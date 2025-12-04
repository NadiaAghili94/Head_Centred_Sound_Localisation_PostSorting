clc;
clear; 
close all;

baseDir_folder = 'E:\Stephen_Nadia_Jenny\Sponge_seprated\Right\Group_01';
csvFilePath    = 'F:\Spontage_results\Right\Sponge_Right__Group_1\recording_info.csv';

data = readtable(csvFilePath, 'PreserveVariableNames', true);
Block_order = data.block;
n_folder = height(data);

Behav_path   = cell(1, n_folder);
onset_times  = cell(1, n_folder);
Behavior_    = cell(1, n_folder);

%  find behaviour (.txt) files
for i_f = 1:n_folder
    subFolderPath = fullfile(baseDir_folder, string(Block_order{i_f}));

    % Find behaviour log
    txtFiles = dir(fullfile(subFolderPath, '*level*.txt'));
    if ~isempty(txtFiles)
        Behav_path{i_f} = fullfile(subFolderPath, txtFiles(1).name);
    else
        Behav_path{i_f} = '';
        fprintf('No behaviour file in %s\n', subFolderPath);
    end
end

%%  extract  behaviour + onset
for i_f = 1:n_folder

    if isempty(Behav_path{i_f}) || ~isfile(Behav_path{i_f})
        fprintf('Skipping %d: no behaviour file\n', i_f);
        continue;
    end

    % Load behaviour
    Behavior_{i_f} = readtable(Behav_path{i_f});
    % Compute onset
    subFolderPath = fullfile(baseDir_folder, string(Block_order{i_f}));
    onset_times{i_f} = align_event_times(subFolderPath);
end
