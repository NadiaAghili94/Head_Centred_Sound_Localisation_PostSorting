function varargout = align_event_times(file_path, opt)
%
%
% 12 Feb 2020 by Stephen Town
%
% Takes start times from behavioral text file (referenced in TDT time
% frame) and finds nearest event on digital event channel representing
% stimulus onset (play now). Does not correct for DAC-ADC latency 
%
% INPUTS
% - file_path: path to directory containing behavioural file and h5 files
% with stimulus sync signal (DOut/Digital Events3)
% - draw (optional): shows alignment of times for visual inspection
%
% OUTPUTS:
% - matched_times: array of stimulus times where time is defined by the
%                  multichannel systems clock
% - trial_index: index to trial for cross-referencing stimulus information

try
           
    % Define folders and drawing options   
    if nargin == 0        
       file_path = uigetdir('E:\UCL_Behaving'); 
    end
    
    if nargin < 2
        opt = struct('draw',true, 'max_accept_delta', 1); %orignal = 0.2
    end
    
    % Get file names
    [h5_files, behav_file] = get_files( file_path);
    
    % Draw start times in behaviour
    B = readtable( fullfile( file_path, behav_file));
    nTrials = size(B, 1);
    
    % Hack for .txt files that won't load properly from early AV
    % integration blocks (RN)
    fprintf('Behavioral file name: %s\n', behav_file);

    % if contains(behav_file, 'level81_SS_Passive')
    % B.Properties.VariableNames = ["Trial", "StartTime", "StimFile", "StimIdx", "VowelF1", ...
    %     "VowelF2", "Pitch", "dB", "Atten", "AttenParam", "Envelope", "Coherence", "vBase", ...
    %     "vScale", "HoldTime", "Blank"];
    % fprintf('Heads up:Behaviour text file column names were not loaded, forcing rename\n')
    % end
    
    if isempty(B)
        matched_times = []; return
    end
    
    % Highlight to user if selecting one of multiple files
    if numel(h5_files) > 1
        fprintf('Heads up: Multiple h5 files detected\n')
       fprintf('Taking events from %s\n', h5_files(1).name)
    end
 
    % Load event times
    H5 = McsHDF5.McsData( fullfile( file_path, h5_files(1).name) );
    stim_obj = get_MCS_digital_events_obj(H5, 'Digital Events3');   
    
    if isempty(stim_obj) 
        error('Could not find stream for Digital Events3 - check integrity of %s', h5_files(1).name)
    end
    
    event_times = double( stim_obj.Events{1}(1,:)); 
    event_times = event_times ./ 1e6;
    
    % Flag if wildly different number of MCS events and behavioural trials
    % This suggests the wrong MCS file might have been converted + moved to block.
    if abs(numel(event_times) - numel(B.StartTime)) > 5
        warning('Number of trials does not match number of events triggered - Check MCS file')
    end
    
    % Align times     
    time_delta = abs( bsxfun(@minus, B.StartTime, event_times));
%     time_delta = abs( bsxfun(@minus, B.Var2 , event_times));      % Rebecca - Use this line (and comment out the line above)
    [min_delta, min_idx] = min(time_delta, [], 2);
    matched_times = event_times(min_idx);    
    
    % Check for missing stimuli 
    matched_times(min_delta > opt.max_accept_delta) = nan;
    n_unmatched = sum( isnan( matched_times));
    
    if  n_unmatched > 0
        warning('Could not match times of %d events', n_unmatched)
    end
    
    % Ouputs
  
     varargout{1} = matched_times;
    % Rebecca updated 
    % varargout{2} = B.Trial';
%     varargout{2} = tranpose(B.Trial);
%     varargout{2} = 1:size(B,1);       % Rebecca - Use this line (and comment out the line above)
    
    
    
    % Optional visualization
    if opt.draw
        figure;
        hold on
        scatter( B.StartTime, zeros( nTrials, 1), 'filled')
        scatter( matched_times, zeros( nTrials, 1)-1, 'filled')
        scatter( event_times, ones(size(event_times)), 'filled')
        set(gca,'ytick', 0 : numel(h5_files),'ylim',[-1 1])
        xlabel('Time (s)')
    end
        
catch err
    % err
    % keyboard
end
    


function [h5_files, behav_file] = get_files( file_path)
    
    h5_files= dir( fullfile( file_path, '*.h5'));
    behav_file = dir( fullfile( file_path, '*Block*.txt'));

    if numel(behav_file) == 0
       error('No behavioral file detected') 
    end

    if numel(behav_file) > 1
       error('Multiple behavioral files detected') 
    else
        behav_file = behav_file(1).name;
    end

    if numel(h5_files) == 0
       error('No neural files detected') 
    end