% CL Septembre 2022, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca

% Toolbox required: Matlab, SPM12


%% 
function Norm=Norm_BMPD(filename_DeformField,filename_func,SPM_Dir)

%______________________________________________________________________
%% Initialization 
%______________________________________________________________________
addpath(SPM_Dir); % Add SPM12 to the path
%f = spm_select('ExtFPList',fullfile(inputDir),'^sub.*\.nii',1) % listes d'images 3D 


matlabbatch{1}.spm.spatial.normalise.write.subj.def = {filename_DeformField};
matlabbatch{1}.spm.spatial.normalise.write.subj.resample = cellstr(filename_func);
matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
NaN NaN NaN];
matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

spm_jobman('initcfg');
spm_jobman('run',matlabbatch);

Norm='Normalisation to MNI space Done'
clear matlabbatch
end   

