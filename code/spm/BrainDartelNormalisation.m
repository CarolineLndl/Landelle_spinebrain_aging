% CL Septembre 2022, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca

% Toolbox required: Matlab, SPM12


%% 
function Norm=Norm_dartel(dartel_template,filename_DeformField,i_file,SPM_Dir,resolution)

%______________________________________________________________________
%% Initialization 
%______________________________________________________________________
addpath(SPM_Dir); % Add SPM12 to the path
%f = spm_select('ExtFPList',fullfile(inputDir),'^sub.*\.nii',1) % listes d'images 3D 

matlabbatch{1}.spm.tools.dartel.mni_norm.template = {dartel_template};
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj.flowfield = {filename_DeformField};
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj.images = cellstr(i_file);
matlabbatch{1}.spm.tools.dartel.mni_norm.vox = str2num(resolution);
matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN
                                               NaN NaN NaN];
matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];
spm_jobman('initcfg');
spm_jobman('run',matlabbatch);

Norm='Normalisation to MNI space Done'
clear matlabbatch
end   

