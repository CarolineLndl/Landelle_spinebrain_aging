%-----------------------------------------------------------------------
% Tapas function for BMPD dataset
% CL Septembre 2020, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca
% Toolbox required: Matlab, SPM12, Tapas PhysiO Toolbox 
% https://github.com/translationalneuromodeling/tapas/tree/master/PhysIO 
% https://www.sciencedirect.com/science/article/pii/S016502701630259X


% %-----------------------------------------------------------------------
%%% outputs
% *.../1_func_denoised/sub-SXX/*  
% .txt => Physiological recordings
% .png => Physiological recordings plots

%%% *.../1_func_denoised/sub-SXX/structure/*   
% .txt => regressor files each column = 1 regressor  
% -- 6 Cardiac regressors (3*(Cos+Sin))  
% --  8 Resp regressors (4*(Cos+Sin))  
% --  4 Interaction regressors (4*1)  
% --  1 heart rate variability (HR) regressor  
% --  1 respiratory volume per time (RV) regressor  

%%% *.../1_func_denoised/sub-SXX/structure/Tapas*  
%  .mat => output from Tapas (c,r,pulse,hv,rvt...)  
% .png => visual output from tapas  
% .txt => regressor files each column = 1 regressor  


% /!\ This function is adapted to siemens recordings acquired without being triggered by fMRI acquisition.
%-----------------------------------------------------------------------
function Tapas=Tapas_BMPD(sub_name, inputDir, func_img, TR, frq_physio,json_file,  nb_slices,  csf_mask, cardio_file, resp_file, trigger_file,outfile_name,outputDir,config)
config=struct(config); % load config parameters
addpath(config.tools_dir.spm_dir) % path for SPM
double(TR)

if json_file==1
frq_physio=[]   
else
frq_physio=[double(1/double(frq_physio))]
end

int64(nb_slices) % number of slices in int64 format
start_acq=0



if config.denoising_params=="slice_wise"
slice_ref=[1:nb_slices] % slice-specific regressors will be computed 
else
    slice_ref=[round(double(nb_slices)/2)];
end
%ROIs{1,1} = csf_mask;
%ROIs{2,1} = wm_mask;
%ROIs
% Files ______________________________________________________________________
% check the file extention:
[filepath,name,ext]=fileparts(cardio_file);

% in case the input file was zipped

 % 'TAPAS PhysIO Toolbox' ================================================================
f = spm_select('ExtFPList',fullfile(inputDir),func_img,Inf); % listes d'images 3D 
ff = fullfile(inputDir,func_img);        % Une image 4D % 4D images

size(f,1)
matlabbatch{1}.spm.tools.physio.save_dir = {outputDir};

% Log files (.tsv or .log files)
if ext=='.tsv' 
    matlabbatch{1}.spm.tools.physio.log_files.vendor = 'BIDS'; %'Siemens'; % Vendor Name depending on your MR Scanner/Physiological recording system
    matlabbatch{1}.spm.tools.physio.log_files.scan_timing = {''};  % last DICOM volume, header have the same time axis as the time stamp in the physiological log file

elseif ext=='.txt' 
    cardio_file
    matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Custom'; %'Siemens'; % Vendor Name depending on your MR Scanner/Physiological recording system

elseif ext=='.log'
    matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Siemens_Tics' %'Siemens'; % Vendor Name depending on your MR Scanner/Physiological recording system
    matlabbatch{1}.spm.tools.physio.log_files.scan_timing = {trigger_file};
elseif ext=='.mat'
    matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Biopac_Mat'; %'Siemens'; % Vendor Name depending on your MR Scanner/Physiological recording system
    matlabbatch{1}.spm.tools.physio.log_files.scan_timing = {resp_file}%{''}; 
end



matlabbatch{1}.spm.tools.physio.log_files.cardiac = {cardio_file};
matlabbatch{1}.spm.tools.physio.log_files.respiration = {resp_file};
  % 
matlabbatch{1}.spm.tools.physio.log_files.sampling_interval = frq_physio; % for Siemens, sampling rate is read directly from logfile, or you can put 1/400 => double check with physio.json
matlabbatch{1}.spm.tools.physio.log_files.relative_start_acquisition = start_acq; %leave this parameter empty or 0 (e.g., since physiological recordings and acquisition timing are already synchronized scan_timing 
matlabbatch{1}.spm.tools.physio.log_files.align_scan = 'last'; % end of logfile will be aligned to last scan volume

% Scan timing
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nslices = double(nb_slices); %'Number of slices in one volume'
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = []; %Only for triggered (gated) sequences
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.TR = TR; % TR should be an integer or double with comma (e.g 2.0)
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Ndummies = 0;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nscans = size(f,1); %'Number of scans (volumes) 
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.onset_slice = slice_ref;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sync.nominal = struct([]);

% Cardiac
matlabbatch{1}.spm.tools.physio.preproc.cardiac.modality = 'PPU';
matlabbatch{1}.spm.tools.physio.preproc.cardiac.filter.no = struct([]);
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0;
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.max_heart_rate_bpm = 90;
matlabbatch{1}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);

%% Outputs
matlabbatch{1}.spm.tools.physio.model.output_multiple_regressors = ['sub-',sub_name,outfile_name,'.txt'];
matlabbatch{1}.spm.tools.physio.model.output_physio = ['sub-',sub_name,outfile_name,'.mat']; 

matlabbatch{1}.spm.tools.physio.model.orthogonalise = 'none';
%matlabbatch{1}.spm.tools.physio.model.censor_unreliable_recording_intervals = true; %values of the nuisance regressors (R-matrix) will be set to zero (=censored) for time points that are in intervals with unreliable related recordings

%% RETROICOR method:
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.c = 3;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.r = 4;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.cr = 1;
matlabbatch{1}.spm.tools.physio.model.rvt.yes.delays = 0;
matlabbatch{1}.spm.tools.physio.model.hrv.yes.delays = 0;
    
%% CompCor method is now running with nypipe
matlabbatch{1}.spm.tools.physio.model.noise_rois.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.movement.no = struct([]);


%nb we have only 2 movement parameters and PhysiO need 3 that wy we
%enter its as 'other multiple regressors'

matlabbatch{1}.spm.tools.physio.verbose.level = 0;
matlabbatch{1}.spm.tools.physio.verbose.fig_output_file = ['sub-',sub_name,outfile_name,'.png']
matlabbatch{1}.spm.tools.physio.verbose.use_tabs = false;


spm_jobman('run',matlabbatch)
Tapas='Tapas Done'


clear matlabbatch
end   