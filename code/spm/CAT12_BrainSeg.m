
% CAT12 for Aging dataset
% CL Septembre 2020, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca

% Toolbox required: Matlab, SPM, CAT12 Toolbox 
% Cat12: http://www.neuro.uni-jena.de/cat/
% Segmentation CAT12 /!\ expert mode:  cat.extopts.expertgui to 1 in cat_defaults.m line 282

% Inputs:------------------- 
% sub_name = name/number of the participant
% outputDir = directory for the outputs
% inputDir = directory for the inputs
% Anat_file = all unzip anat with cropped brain

%% 
function CAT=CAT_SPPark(filename_anat,workingDir,SPM_Dir)

%______________________________________________________________________
%% Initialization 
%______________________________________________________________________
SPM_Dir=SPM_Dir;
addpath(SPM_Dir) % Add SPM12 to the path
mkdir(workingDir);
cd([ workingDir ]) ; % create .txt files during processing in a working directory
Anat_All ={filename_anat} ;% create variable that containt all anat    

matlabbatch{1}.spm.tools.cat.estwrite.data = Anat_All; % anat for all subjects
matlabbatch{1}.spm.tools.cat.estwrite.nproc = 6; % Split job into separate processes

    
%______________________________________________________________________
%% Options for initial SPM 12 preprocessing 
%______________________________________________________________________
matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {[SPM_Dir '/tpm/TPM.nii,1']}; % tissue probability maps
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr = 0.5; % inhomogeneity correction
matlabbatch{1}.spm.tools.cat.estwrite.opts.accstr = 0.5; % accuracy of preprocessing function 0.5 = average, 0.75: high, 1: ultra high

%______________________________________________________________________
%% Extended options for CAT12 preprocessing : Segementation options 
%______________________________________________________________________
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070; % Affine PreProcess, estimate intensity inhomogenity
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.NCstr = -Inf; % Strength of Noise Corrections
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.LASstr = 0.5; % Strengh of local Adaptive Segmentation (ie. basal ganglia, M1...) 0.5=medium
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.gcutstr = 0; % Skull-Stripping; 0=SPM approach
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.cleanupstr = 0.5; % Strengh of final Clean Up (removes meninges  & partial volume effects) 0.5=medium
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.WMHC = 1; % WM hyperintensity Correction (for aging) 1: yes
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.SLC = 0; % Stroke Lesion correction 0= no
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.restypes.fixed = [1 0.1]; % isotropic voxel size is controlled by the first parameter
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm = {[SPM_Dir 'toolbox/cat12/templates_MNI152NLin2009cAsym/Template_1_Dartel.nii,1']}; % 1rst image of the Dartel template
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.shootingtpm ={[SPM_Dir 'toolbox/cat12/templates_MNI152NLin2009cAsym/Template_0_GS.nii']}; V1
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.regstr = 0; V1
matlabbatch{1}.spm.tools.cat.estwrite.extopts.vox = 1.5; % voxel size of normalised images should be the same as the template /!\

%______________________________________________________________________
%%  Extended options for CAT12 preprocessing : Surface options 
%______________________________________________________________________
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.pbtres = 0.5; % Voxel size for thickness estimation
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.scale_cortex = 0.7; % Intensity value for Cortical surface creation (GM/WM border, gyri/sulci)
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.add_parahipp = 0.1; % Modify parahippocampal surface (prevent large cuts in this region)
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.close_parahipp = 0; 


%______________________________________________________________________
% Extended options for CAT12 preprocessing : Administration options 
%______________________________________________________________________
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.ignoreErrors = 0; % 0: ignore errors; 1: catch errors
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.verb = 2; % verbose processing level 2: details
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.print = 2; % Create final CAT report (2: yes volume only)

%______________________________________________________________________
% Writing options ----------------------------------------
%______________________________________________________________________

matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 1; % surface & thickness estimation 0= no
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.neuromorphometrics = 0; %
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.lpba40 = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.cobra = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.hammers = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.ibsr = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.aal = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.mori = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.anatomy = 0;
%matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_200Parcels_7Networks_order = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.ownatlas = {[SPM_Dir 'toolbox/cat12/templates_MNI152NLin2009cAsym/Schaefer2018_200Parcels_7Networks_order.nii,1']};

matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.julichbrain = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.atlases.suit = 1;



matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native = 1; % create native image 'p*.nii'
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.warped = 0; % normalized without any modulation 0: no
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 0; % modulated normalized : 0:no
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.dartel = 2; % Dartel export
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.dartel = 2;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.native = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.dartel = 2;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.atlas.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.atlas.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.label.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.label.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.label.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.bias.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.bias.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.bias.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.las.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.las.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.las.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.jacobianwarped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.warps = [1 1]; % save deformation field  [1 1]: forward + inverse

spm_jobman('initcfg');
spm_jobman('run',matlabbatch);

CAT='Segmentation Done'
clear matlabbatch
end   

