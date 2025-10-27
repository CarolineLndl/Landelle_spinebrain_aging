import pandas as pd
import nibabel as nib
import numpy as np
import os.path
import os, glob
from joblib import Parallel, delayed
import time
from scipy.stats import zscore, iqr
import brsc_utils as util

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Nipype
from nipype.interfaces.fsl.maths import StdImage
from nipype.interfaces.fsl import ImageStats

# Nilearn
from nilearn.image import math_img, smooth_img
from nilearn.masking import apply_mask

from brsc_preprocess import Preprocess_Sc

from scipy import signal

class ALFF:
    '''
    The StaticFC class is used to compute and visualize sFC
    metrics on functional timecourses
        - connectivity analyses
        - ALFF analyses

    Attributes
    ----------
    config : dict
        Contains information regarding IDjects, runs, rois, etc.
    '''

    def __init__(self, config,IDs,config_proc=None,analysis='alff', structure='spinalcord'):
        '''
        Parameters
        ----------
        config : dict
            Contains information regarding IDjects, runs, rois, etc.
        IDs : list
            List of IDs to be analyzed
        config_proc : dict
            Contains information regarding preprocessing parameters, only used if space is 'indiv_space'
        analysis : str
            Type of analysis to be performed (default is 'alff')
        structure : str
            Structure to be analyzed (default is 'spinalcord')
        '''

        self.config = config
        self.analysis=analysis
        self.IDs=IDs
        self.structure=structure
        self.population_info=self.config["project_dir"] +config["population_infos"]

        if config_proc is not None:
            self.config_proc = config_proc
            self.preprocess_Sc=Preprocess_Sc(self.config_proc)

        # Extrat the analysis parameters from the config file
        freq_label = None  # Identify frequency band label based on config

        for label, frq in self.config["alff"]['frq_range'].items():
            if frq == self.config["alff"]['alff_freq_range']:
                freq_label = label
                break
        
        if freq_label is None:
            raise ValueError("Provided frequency range does not match predefined options in the config file.")

        # define output directory and create if it does not exist
        self.output_dir = config["project_dir"] + config["alff"]["analysis_dir"][structure] + '/' +self.analysis+'/'+ freq_label + '/'
        print(f'Output directory: {self.output_dir}')
       
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(self.output_dir + "/1_first_level/")
            os.mkdir(self.output_dir + "/plots/")
        
                   
    def compute_alff_maps(self,space='template_space',fwhm=None,normalization=False,scaling_method="zscore",redo=False,verbose=100,n_jobs=1):
        '''Computes amplitude of low frequency fluctuations (ALFF) for each voxel
        (i.e., sum of the amplitudes in the low frequency band, as defined in config file)


        Inputs
        ----------
        space : str
            Space in which the ALFF should be computed (default is template_space)
        fwhm : float
            Full width at half maximum for smoothing (default is None)
        normalization : bool
            Set to True to normalize the ALFF maps
        scaling_method : str
            Method to scale the ALFF maps, either 'zscore' or 'robust_sigmoid' (default is 'zscore')
        redo : bool
            Set to True to overwrite existing files
        n_jobs : int
            Number of parallel jobs to run (default is 1)


        Outputs
        ----------
        yourtimeseries_alff.nii.gz
            ALFF image (i.e., std)
        yourtimeseries_alff_Z.nii.gz
            Z-scored ALFF image 

        '''
        print('ALFF MAPS COMPUTATION')
        print(f'Overwrite old files: {redo}')

        print('... Preparing data')
        # Linearize list of IDjects & runs to run in parallel
        all_mask=[]
        all_ID = []  # This will contain all the paths without extension, so that they suffixes can be added later on
        all_ID_smoothed = []
        ID_ouput_dir=[]

        if not os.path.exists(self.output_dir + "/1_first_level/nifti/"):
            os.makedirs(self.output_dir + "/1_first_level/nifti/")

        for ID in self.IDs:
            preproc_dir= self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
            ID_func_file = glob.glob(self.config["project_dir"] + self.config["alff"][space][self.structure]['func'].format(ID))[0]
            
            all_ID.append(ID_func_file) # Add to list of IDjects, not smoothed
            #all_ID_smoothed.append(glob.glob(preproc_dir + self.config[space][self.structure]['func_smooth'].format(ID))[0])
            if space=="indiv_space":
                all_mask.append(glob.glob(preproc_dir +self.config[space][self.structure]['func_mask'].format(ID))[0])
            elif space=="template_space":
                all_mask.append(glob.glob(self.config["project_dir"] +self.config["alff"][space][self.structure]['func_mask'])[0])

        output_dir=self.output_dir + "/1_first_level/nifti/"
        print('... '+self.analysis+' is Running')
        start = time.time()
        files=Parallel(n_jobs=n_jobs,
                 verbose=verbose,
                 backend='loky')(delayed(self._alff)(all_ID[file],output_dir,all_mask[file],fwhm,scaling_method,redo)
                                   for file in range(0,len(all_ID)))
        print("... Operation performed in %.3f s" % (time.time() - start))
        alff_f, z_alff_f=zip(*files)

    ### Normalization
        if space == 'indiv_space' and normalization == True:
            # Normalize the ALFF maps to template space
            if self.structure == 'spinalcord':
                print('... Normalizing to template space')
            
            o_folder=[];warp_f=[];o_files=[];dest_img=[]
            
            for ID_nb, ID in enumerate(self.IDs):
                preproc_dir=self.config_proc["preproc_dir_bmpd"] if ID[0]=="P" else self.config_proc["preproc_dir"]
                o_folder.append(output_dir)
                warp_f.append(glob.glob(preproc_dir + self.config["template_space"][self.structure]["func_warp"].format(ID))[0])
                o_files.append(z_alff_f[ID_nb].replace('.nii.gz','_inTemplate.nii.gz'))
                dest_img.append(self.config["project_dir"]+ self.config["templates"]["spinalcord"]["t2"])
            
            self.preprocess_Sc.apply_warp(i_img=z_alff_f,
                ID=self.IDs,
                o_folder=o_folder,
                dest_img=dest_img,
                warping_field=warp_f,
                tag="_inTemplate",
                n_jobs=n_jobs,redo=redo)
            
            # mask the ALFF maps in Template space
            for file in o_files:
                if self.structure == 'spinalcord':
                    mask=self.config['project_dir']+self.config['templates'][self.structure]["cord"]
                str_mask = f"fslmaths {file} -mas {mask} {file}"
                os.system(str_mask)
     
        print('...DONE!')
        self.z_alff_f=z_alff_f
        self.alff_f=alff_f

        return alff_f, z_alff_f

        
    def extract_alff_rois(self, input_f=None,atlas_f=None,atlas_labels=None,atlas_labels_nb=None,space="template_space",tag="",redo=False,verbose=1,plot=True):
        '''Extract ALFF values in specific ROIs and put them into a dataframe
        '''

        # Initialize the function
        print(f'ALFF IN ROIS')
        if redo==True:
            print(f'Overwritting old files ')

        if not os.path.exists(self.output_dir + "/1_first_level/metric/"):
            os.mkdir(self.output_dir + "/1_first_level/metric/")
        ana_tag = 'alff' + tag if self.analysis=="alff" else 'falff' + tag
        
        # select atlas files
        if atlas_f is None and space=="template_space":
            atlas_f=glob.glob(self.config['project_dir'] + self.config['templates'][self.structure]['atlas'])[0]
        
        elif atlas_f is None and space=="indiv_space":
            atlas_f=[]
            
            for ID_nb, ID in enumerate(self.IDs):
                preproc_dir=self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
                atlas_f.append(glob.glob(preproc_dir + self.config["indiv_space"][self.structure]["func_atlas"].format(ID))[0])
        if atlas_labels is None:
            raise Warning("Atlas labels should be provided, you can for instance provide a list of numbers from 1 to n")
                
        # Select the input data files
        masked_data_list = []
        if input_f is None:
            data_files =self.alff_f           
        else:
            data_files = input_f


        # Loop over the IDs and extract the ALFF values and put them into a dataframe

        all_data=[]
        for ID_nb, ID in enumerate(self.IDs):
            if not os.path.exists(self.output_dir + "/1_first_level/metric/sub-"+ID+"_"+ana_tag+".csv") or redo:  
                  
                data_file=data_files[ID_nb]
                data_img = nib.load(data_file)
                data_data = data_img.get_fdata()
                

                IDs_region_mean=[]
                IDs = [] ; rois = [] ; groups = [] ; ages=[]; sex=[]
                for label_nb, label in enumerate(atlas_labels):
                    
                    if atlas_labels_nb is not None:
                        label_idx=atlas_labels_nb[label_nb]
                    else:
                        label_idx=label_nb+1

                    metadata = pd.read_csv(self.population_info, delimiter='\t')
                    IDs.append(ID)
                    groups.append(metadata [metadata ["participant_id"] == ID]["group"].values[0])
                    ages.append(metadata [metadata ["participant_id"] == ID]["age"].values[0])
                    sex.append(metadata [metadata ["participant_id"] == ID]["sex"].values[0])
                    rois.append(label)
        
                    # extract the ALFF values in the ROIs

                    if space=="template_space":
                        atlas_img = nib.load(atlas_f)
                            
                    elif space=="indiv_space":
                        atlas_img =  nib.load(atlas_f[ID_nb])
                    
                    
                    atlas_data= atlas_img.get_fdata()

                    mask=atlas_data==int(label_idx)
                    values=data_data[mask]

                    
                    
                    IDs_region_mean.append(np.mean(values))

                    ID_values=np.array(IDs_region_mean)
                    #if len(atlas_labels) > 1:
                        #ID_values=zscore(ID_values, nan_policy='omit') # Normalize by the mean of the ID
                        
                #create the individual dataframe
                colnames = ["IDs","age","sex","groups","rois","alff"]
                alffs = pd.DataFrame(list(zip(IDs, ages,sex,groups, rois, ID_values)), columns=colnames)


                if self.config["labels1"] and self.structure=="spinalcord":
                    alffs['ventro_dorsal'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels_VD"]), axis=1)
                    alffs['right_left'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels_RL"]), axis=1)

                    alffs['levels'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["level_labels"]), axis=1)
               
                elif self.config["labels1_brain"] and self.structure=="brain":
                    alffs['right_left'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels_RL_brain"]), axis=1)
                    alffs['ventro_dorsal'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels3_brain"]), axis=1)
                    alffs['structure'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels1_brain"]), axis=1)
                    alffs['networks'] = alffs.apply(lambda row: util.assign_labels1(row['rois'], self.config["network_labels"]), axis=1)



                #save as csv
                alffs.to_csv(f"{self.output_dir}/1_first_level/metric/sub-{ID}_{ana_tag}.csv",index=False)


              
            else:
                if verbose>0:
                    print('ALFFs already extracted, loading from .csv file...')
                alffs=pd.read_csv(f"{self.output_dir}/1_first_level/metric/sub-{ID}_{ana_tag}.csv")

            all_data.append(alffs)

        # Concatenate all the individual dataframes into a single dataframe
        all_alffs = pd.concat(all_data, ignore_index=True)
        os.makedirs(self.output_dir + "/2_second_level/metric/", exist_ok=True)
        
        output_group_file = self.output_dir + "/2_second_level/metric/n"+str(len(self.IDs))+"_"+ana_tag+".csv"
        if not os.path.exists(output_group_file) or redo:
            # Save the combined dataframe to a new CSV file
            all_alffs.to_csv(output_group_file, index=False)

        if plot == True:
            plt.figure(figsize=(10,6))
            sns.barplot(x="roi",y=ana_tag,hue="group",data=all_alffs,palette='flare')       
            plt.savefig(f"{self.output_dir}/plots/{ana_tag}_rois.pdf")
    
        return all_alffs
        
        

    # Utilities
    def _alff(self,data_main_dir,output_dir,mask_f,fwhm,scaling_method,redo):
        '''Compute alff map for one IDject
        The preprocessed timeseries should not be standardized, but the ALFF map will be standardized

        Inputs
        ----------
        data_main_dir : str
            Path to the image to flip (i.e., no suffix, no file extension)
        
        Outputs
        ----------
        data_main_dir_alff_pam50.nii.gz
            ALFF image (i.e., std)
        data_main_dir_alff_Z_pam50.nii.gz
            Z-scored ALFF image
        '''
        output_tag = '_alff' if self.analysis=="alff" else '_falff'
     
        #Compute ALFF or fALFF
        alff_f=output_dir + os.path.basename(data_main_dir).split('.')[0] +output_tag+'.nii.gz'
        if (not os.path.isfile(output_dir + os.path.basename(data_main_dir).split('.')[0] +output_tag+'.nii.gz')) or redo :#or ((output_dir +os.path.basename(data_main_dir).split('.')[0]  +'_alff_Z.nii.gz')) or self.config['overwrite_alff_maps']: # If not already done or if we want to overwrite
            
            if os.path.isfile(data_main_dir):
                # Load the data and define the frequency range of interest
                img = nib.load(data_main_dir)
                data = img.get_fdata()
                N = data.shape[-1]  # Number of time points
                if mask_f:
                    mask_img = nib.load(mask_f)
                    mask = mask_img.get_fdata() > 0  # Convert to boolean mask
                    data[~mask] = np.nan  # Set non-masked regions to NaN

                lowcut, highcut = self.config['alff']['alff_freq_range'] # Frequency range for ALFF computation
                
                # Step 1: detrend the data
                masked_data = np.all(np.isfinite(data), axis=-1)
                data_detrended = np.full_like(data, np.nan)
                data_detrended[masked_data, :] = signal.detrend(data[masked_data, :], axis=-1)

                # Step 2: Perform Fourier Transform along the time axis
                fft_result = np.fft.fft(data_detrended, axis=-1)  
                amplitude_spectrum = np.abs(fft_result)  # Amplitude spectrum
                power_spectrum = (amplitude_spectrum ** 2)   # Power spectrum is obtained by squaring the amplitude spectrum


                #Step3: Define frequency bins based on TR
                freqs = np.fft.fftfreq(N, d=self.config['alff']['TR']) # Compute frequency bins based on TR
                f_mask_bp = np.logical_and(freqs >= lowcut, freqs <= highcut)  # Band-pass mask

                #Step4: Compute ALFF (square root of power spectrum, then averaged in low-freq range) 
                alff_map = np.sqrt(np.nanmean(power_spectrum[..., f_mask_bp], axis=-1))
                #alff_map = np.nan_to_num(alff_map, nan=0.0, posinf=0.0, neginf=0.0)# # Handle NaN and Inf values

                # Step 5: Save the ALFF image
                if self.analysis=='alff':
                    img_alff = nib.Nifti1Image(alff_map, img.affine, img.header)
                    nib.save(img_alff, output_dir + os.path.basename(data_main_dir).split('.')[0]  +output_tag+'.nii.gz')

                if self.analysis=='falff':
                    # Step 6: Compute total amplitude power (for fALFF) and save image
                    f_mask_all = np.logical_and(freqs >= 0, freqs <= 0.25)  # Mask for all positive frequencies
                    total_power = np.sum(power_spectrum[..., f_mask_all], axis=-1) # denominator for fALFF
                    band_power = np.sum(power_spectrum[..., f_mask_bp], axis=-1) # numerator for fALFF

                    #falff_map = np.divide(alff_map, total_power, out=np.zeros_like(alff_map), where=total_power != 0)
                    #falff_map = np.divide(band_power, total_power, out=np.zeros_like(alff_map), where=total_power != 0)
                    falff_map = np.divide(band_power, total_power)

                    img_falff = nib.Nifti1Image(falff_map, img.affine, img.header)
                    nib.save(img_falff, output_dir + os.path.basename(data_main_dir).split('.')[0]  +output_tag+'.nii.gz')



                #clean the alff values by removing the outliers
                #mean_alff = np.mean(alff[alff > 0])  # Exclude zeros
                #std_alff = np.std(alff[alff > 0])  # Exclude zeros
                #std_threshold = 2  # std threshold
                #threshold_value = mean_alff - (std_threshold * std_alff) # Create a mask to exclude values below mean - (std_threshold * std)
                #mask_low_std = alff >= threshold_value  # Keep values above threshold
                #alff = alff * mask_low_std# Apply the mask to remove low ALFF values

            else:
                raise Exception(f'Input file {data_main_dir}.nii.gz does not exist.')   
              
        # post- processing of alff images
        if scaling_method=="zscore":
            z_tag="_Z"
        elif scaling_method=="robust_sigmoid":
            z_tag="_sig"

        z_alff_f=output_dir + os.path.basename(data_main_dir).split('.')[0] + output_tag+ '_s'+z_tag+'.nii.gz'
        if not os.path.exists(z_alff_f) or redo:
            ###Smooth image
            alff_f=output_dir + os.path.basename(data_main_dir).split('.')[0] + output_tag+".nii.gz"
            alff_smoothed=output_dir + os.path.basename(data_main_dir).split('.')[0] + output_tag+'_s.nii.gz'
            smoothed_image=smooth_img(imgs=alff_f,
            fwhm=fwhm)
            smoothed_image.to_filename(alff_smoothed)

            ### compute alff Z-score
            data = nib.load(alff_smoothed).get_fdata()
            masked_data = data[~np.isnan(data)]  # Keep only finite values
            mean = np.mean(masked_data)
            std = np.std(masked_data)
            med = np.nanmedian(masked_data)
            scale = iqr(masked_data, nan_policy='omit') / 1.35  # Approximate std
            print(scale)
            z_alff_f = output_dir + os.path.basename(data_main_dir).split('.')[0] + output_tag+ '_s'+z_tag+'.nii.gz'
            alff_img = nib.load(alff_f)#'_masked.nii.gz')
            if scaling_method=="zscore":
                z_alff_img = math_img("(img - {}) / {}".format(mean, std), img=alff_img)

            elif scaling_method=="robust_sigmoid":
                z_alff_img = math_img("1 / (1 + np.exp(-(img - {}) / {}))".format(med, scale), img=alff_img)



            z_alff_img.to_filename(z_alff_f) #'_masked_Z.nii.gz')
            os.remove(alff_smoothed)


            # change nan in the image to 0
            for img in [alff_f, z_alff_f]:
                str_mask = 'fslmaths ' + img+ ' -nan '+  img
                os.system(str_mask)

            
        return alff_f, z_alff_f
    

        '''
        Extracts csa
        '''
        
        if not os.path.exists(o_file) or redo==True:
            if method=="wa":
                string="sct_extract_metric -i " + i_img +" -method " +method+  " -f " + mask_path + " -o "+o_file+ " -vert " + levels + " -vertfile " + level_img + " -perlevel 1" 
            else :
                string="sct_extract_metric -i " + i_img +" -method " +method+  " -f " + mask_path + " -l " + labels+ " -o "+o_file+ " -vert " + levels + " -vertfile " + level_img + " -perlevel 1" 
                
            os.system(string)
            print(metric_tag + " was extracted for sub-" + ID)
        
        #elif verbose==True:
                #print("CSA was already computed for sub-" + ID)

        return o_file