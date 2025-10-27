# -*- coding: utf-8 -*-
import glob, os, shutil, json, scipy, math
import nibabel as nib
import numpy as np
import pingouin as pg
import pandas as pd
import brsc_utils as util

# Time computation libraries
from joblib import Parallel, delayed
import time
from tqdm import tqdm

# Nilearn library
from nilearn.maskers import NiftiMasker
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import image

# Sklearn library
from sklearn.feature_selection import mutual_info_regression
from sklearn import decomposition
from sklearn.linear_model import Lasso



# Statistics library
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from dtaidistance import dtw
from scipy.stats import zscore

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

class Seed2target:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, config_file,IDs=None,kind='corr',seed_kind="atlas",ana='seed2seed',structure=["spinalcord"],space="template_space",verbose=1):
        '''
        kind: 'corr' or 'cov' or 'pcorr'
            Type of connectivity metric: correlation or covariance or partial correlation

        ana: "seed2seed" or "seed2voxels"
            if seed_kind=="seed2seed", seed_kind is important 
            if seed2voxels, seed_kind is going to be taken into account but structure should have 2 values eg: ['spinalcord','brain'], the first one is going to be the seeds location and the other the target
        
        seed_kind: "atlas" or "rois"
            if seed_kind=="atlas" atlas folder should be provided for config["seeds"]["atlas_dir"], the folder should contain a 3D image with one value for each rois 
            and a .txt file with info about the rois labels. No information is needed for "targets". structure can be "brain" or "spinalcord" or "brain_spinalcord"
            if seed_kind =="rois" distinct rois can be used for seeds and targets,  the mask directory ["seeds"]["dir"] or ["targets"]["dir"] 
            + the names of the masks ["seeds"]["names"] or ["targets"]["names"] and the structure (either "brain" or "spinalcord")

        

        To implement: load different mask_seeds and mask_targets for each individual
        '''
        # '''Load configuration from a JSON file.'''
        with open(config_file) as config_f:
            self.config = json.load(config_f) 
        self.config_file=config_file

        self.structure=structure
        self.ana=ana
        self.seed_kind=seed_kind
        self.IDs= self.config["participants_IDs"] if IDs==None else IDs
        self.kind=kind

        #'''Set up necessary directories based on the configuration.''
         
        if len(self.structure)==1 and ana=='seed2voxels':
            raise Warning("The structure argument should contain two values. For example, structure=['spinalcord', 'brain'] or structure=['spinalcord', 'spinalcord']. The first value represents the location of the seeds, and the second value represents the location of the voxels.")

        elif len(self.structure)>1:
            self.outputdir= self.config["project_dir"] +self.config[ana]["analysis_dir"][self.structure[0]].split(self.structure[0])[0] +self.structure[0]+   "_"+self.structure[1]
            self.target_structure=self.structure[1]
        
        else:
            self.outputdir= self.config["project_dir"] +self.config[ana]["analysis_dir"][self.structure[0]]
            self.target_structure=self.structure[0] # the seed is always the first structure

        if verbose==1:
            print("Your are going to run "+ self.ana + " analysis here:")
            print(self.outputdir)
            print("")

        if ana=='seed2voxels':
            seed_kind='rois'

        self.seed_structure=self.structure[0] # the seed is always the first structure
        
        if ana=='seed2seed' and verbose==1:
            print(f"Seeds are located within the {self.seed_structure} structure")
        if ana=='seed2voxels' and verbose==1:
            print(f"Seeds are located within the {self.seed_structure} structure and voxels withing the {self.target_structure} structure ")
        

        self.firstlevel_dir=self.outputdir + "/1_first_level/"
        self.secondlevel_dir=os.path.join(self.outputdir +"/2_second_level/" + self.kind +'/')
        #self.seed_names=self.config[ana]["seeds"][self.seed_structure]["names"]
        #self.target_names=self.config["targets"][self.target_structure]["names"]
        #''' create output directory if needed '''
        
        
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
            os.mkdir(self.firstlevel_dir)
            os.makedirs(self.secondlevel_dir)
        
        if seed_kind=="rois":
            for target_name in self.target_names:
                
                if not os.path.exists(self.firstlevel_dir+target_name ):
                    
                    os.mkdir(self.firstlevel_dir+target_name)
                    os.mkdir(self.firstlevel_dir+target_name +'/timeseries/') # folder to store timeseries extraction
                    os.mkdir(self.firstlevel_dir+target_name +'/' + target_name  +'_fc_maps/') # folder to store maps of FC
            
            for seed_name in self.seed_names:
                
                for target_name in self.target_names:
                    if not os.path.exists(self.firstlevel_dir+seed_name):
                        os.mkdir(self.firstlevel_dir+seed_name)
                        os.makedirs(self.firstlevel_dir+seed_name+'/timeseries/') # folder to store timeseries extraction
                    if not os.path.exists(self.firstlevel_dir+seed_name+'/'+ target_name +'_fc_maps/'):
                        os.mkdir(self.firstlevel_dir+seed_name+'/'+ target_name +'_fc_maps/') # folder to store maps of FC

            if not os.path.exists(self.secondlevel_dir+seed_name ):
                
                os.makedirs(self.secondlevel_dir+seed_name +'/corr_values/') # folder to store timeseries extraction
                os.makedirs(self.secondlevel_dir+seed_name +'/fc_maps/') # folder to store maps of FC

            for directory in [self.firstlevel_dir+"/timeseries/", self.secondlevel_dir]:
                os.makedirs(directory, exist_ok=True)
        
        if seed_kind=="atlas":
            os.makedirs(self.firstlevel_dir+'/timeseries/', exist_ok=True)
        
        #'''Select masks or atlas files based on the configuration.'''
        print("")
        
        if seed_kind=="rois":
            self.mask_seeds = {}; self.mask_targets = {}
            for seed_nb, seed_name in enumerate(self.seed_names):
                self.mask_seeds[seed_name]=[]
                for ID in self.config['participants_IDs']:
                    ID='sub-' +  ID
                    self.mask_seeds[seed_name].append(glob.glob(self.config["project_dir"] + self.config["seeds"][self.seed_structure]["dir"]+ seed_name + ".nii.gz")[0]) # mask of the seed for the analysis
                
                print("seed #" + str(seed_nb+1) + ": " +seed_name )
        
            print("")
            for target_nb, target_name in enumerate(self.target_names):
                #print(self.config["project_dir"] + self.config["targets"]["target_dir"]+ target_name  + ".nii*")
                self.mask_targets[target_name]=[]
                #print(self.config["project_dir"] + self.config["targets"][self.target_structure]["dir"]+ target_name  + ".nii*")
                self.mask_targets[target_name].append(glob.glob(self.config["project_dir"] + self.config["targets"][self.target_structure]["dir"]+ target_name  + ".nii*")[0]) # mask of the target for the analysis
                
                if verbose==1:
                    print("target #" + str(target_nb+1) + ": " +target_name )
        

        if seed_kind=="atlas":
            self.atlas_img = None; self.atlas_labels = None
            self.atlas_img=glob.glob(self.config["project_dir"] + self.config[ana]["seeds"][self.seed_structure]["atlas_dir"]+ self.config[ana]["seeds"][self.seed_structure]["atlas_tag"] +"/*.nii*")[0]
            if len(self.structure) > 0:
                self.atlas_img_target=glob.glob(self.config["project_dir"] + self.config[ana]["seeds"][self.target_structure]["atlas_dir"]+ self.config[ana]["seeds"][self.target_structure]["atlas_tag"] +"/*.nii*")[0]
            
            
            #print(self.seed_structure +" atlas image: " + os.path.basename(self.atlas_img))
            self.atlas_labels=glob.glob(self.config["project_dir"] + self.config[ana]["seeds"][self.seed_structure]["atlas_dir"]+ self.config[ana]["seeds"][self.seed_structure]["atlas_tag"] +"/"+self.config[ana]["seeds"][self.seed_structure]["atlas_tag"] +".txt")[0]
            if verbose==1:
                print(self.config["project_dir"] + self.config[ana]["seeds"][self.seed_structure]["atlas_dir"]+ self.config[ana]["seeds"][self.seed_structure]["atlas_tag"] +"/"+self.config[ana]["seeds"][self.seed_structure]["atlas_tag"] +".txt")
                print(self.seed_structure +" atlas info: " + os.path.basename(self.atlas_labels))
                print("")

           

        #>>> Select functional data: -------------------------------------
        if verbose ==1:
            print("")
            print("Start the analysis on: " + str(len(self.IDs))+ " participants")
        self.data_seed=[];self.data_target=[]
        for ID in self.IDs:
            if seed_kind=="rois":
                # images selected for extraction:
                #print(self.config['project_dir'] + self.config["input_func"]["seed_dir"] + ID +'/'+ self.seed_structure +'/*'+ config["input_func"]["seed_tag"] +'*')
                #print(self.config['project_dir']+ self.config["input_func"]["target_dir"].format(ID,self.target_structure) +'/*'+ self.config["input_func"]["target_tag"] +'*')
                self.data_seed.append(glob.glob(self.config['project_dir'] + self.config["input_func"]["seed_dir"].format(ID,self.seed_structure) +'/*'+ self.config["input_func"]["seed_tag"] +'*')[0])
                self.data_target.append(glob.glob(self.config['project_dir']+ self.config["input_func"]["target_dir"].format(ID,self.target_structure) +'/*'+ self.config["input_func"]["target_tag"] +'*')[0])
                    
            if seed_kind=="atlas":
                structure_tag = "brain" if structure in ["cortex", "subcortex", "brain"] else self.structure[0]
                #print(self.config['main_dir'] + self.config[space][self.seed_structure]["func"].format(ID,self.seed_structure))
                self.data_seed.append(glob.glob(self.config['project_dir'] + self.config[self.ana][space][self.seed_structure]["func"].format(ID,self.seed_structure))[0])
                self.data_target.append(glob.glob(self.config['project_dir']+ self.config[self.ana][space][self.target_structure]["func"].format(ID,self.target_structure))[0])
             
                labelslist=np.genfromtxt(self.atlas_labels, usecols=1,skip_header=1, dtype="S", delimiter="\t", encoding=None)
                labelslist = np.array([label.decode("utf-8") for label in labelslist])
                labelsnb=np.genfromtxt(self.atlas_labels, usecols=0,skip_header=1, dtype="S", delimiter="\t", encoding=None)
                labelsnb = np.array([label.decode("utf-8") for label in labelsnb])
                    
    
                self.seed_names=labelslist
                

                if len(self.structure) > 1:
                    self.atlas_labels_target=glob.glob(self.config["project_dir"] + self.config[ana]["seeds"][self.target_structure]["atlas_dir"]+ self.config[ana]["seeds"][self.target_structure]["atlas_tag"] +"/"+self.config[ana]["seeds"][self.target_structure]["atlas_tag"] +".txt")[0]
                    labelslist=np.genfromtxt(self.atlas_labels_target, usecols=1,skip_header=1, dtype="S", delimiter="\t", encoding=None)
                    labelslist = np.array([label.decode("utf-8") for label in labelslist])
                    labelsnb=np.genfromtxt(self.atlas_labels_target, usecols=0,skip_header=1, dtype="S", delimiter="\t", encoding=None)
                    labelsnb = np.array([label.decode("utf-8") for label in labelsnb])
                    
                    
                    self.target_names=labelslist
                
                else:
                    self.target_names=labelslist
                
                self.lut=pd.DataFrame({'index': labelsnb,'name': labelslist})


    def extract_atlas_data(self,rmv_col=None,smoothing=None,standardize=False,redo=False,n_jobs=1,verbose=1):
        '''
            Extracts time series within mask (atlas) that there are contained in a 3d MRI file:
            in this image a single region is defined as the set of all the voxels that have a common label (i.e. same value)
            Inputs
            ----------
            rmv_col: column in the label file that contain 0,1 values 0: keep this roi ; 1: remove this roi
            
            smoothing: array
                to apply smoothing during the extraction (ex: [6,6,6]) or set None for no smoothing
                
            redo: 
                if True the extraction will be rerun else if the timeseries were already extracted, the file containing the data will be loaded
            n_jobs: 
                Number of jobs for parallelization
                
            Returns
            ----------
            timeseries_dict: 
                timeseries of each voxels for each participant for the seeds and the targets
                timeseries['seeds']={'raw':[],'zscored':[],'mean':[],'zmean':[],'PC1':[]}"
                timeseries['targets']={'raw':[],'zscored':[],'mean':[],'zmean':[],'PC1':[]}"

            
        '''
        
        # Initiate the extraction for each structure (e.g: 'spinalcord' or 'brain')
        ts_all={}
        atlas_img=[]

        for structure_nb, structure in enumerate(self.structure):
            if structure==self.seed_structure:
                data=self.data_seed
                atlas_img.append(self.atlas_img)
                firstlevel_dir=self.firstlevel_dir
                
            if self.seed_structure != self.target_structure:

                if structure==self.target_structure:
                    data=self.data_target
                    atlas_img.append(self.atlas_img_target)
                
                firstlevel_dir=self.outputdir + "/1_first_level/"

           
            # Initiate timeseries output text file:
            ts_txt=[] # empty array
            for ID_nb, ID in enumerate(self.IDs):
                ts_txt.append(firstlevel_dir+'/timeseries/sub_' + ID + '_' + structure+ '_timeseries') # output filename
            #print(ts_txt)
            # run the extraction in parallel jobs
            if verbose==1:
                if os.path.exists(ts_txt[0] + "_zmean.npy") and redo==False:
                    print("The timeseries were already extracted, loading them ...")
                else:
                    print("Extract time serie within :" + atlas_img[structure_nb])
            ts_all[structure]=Parallel(n_jobs=n_jobs)(delayed(self._extract_ts_atlas)(atlas_img[structure_nb],data[ID_nb],ts_txt[ID_nb],redo,smoothing,structure,standardize)
                                       for ID_nb in range(len(self.IDs)))
        
        # Copy the config file in the output directory
        with open(os.path.dirname(ts_txt[0]) + os.path.basename(self.config_file), 'w') as fp:
                json.dump(self.config, fp)

        # Organize the ts_all metrics in a dictionnary:
        timeseries={}; timeseries_labels={}; labels_all=[]
        for structure_nb, structure in enumerate(self.structure):
            timeseries[structure]={"zmean":[]}
            for ID_nb in range(len(self.IDs)):
                timeseries[structure]["zmean"].append(ts_all[structure][ID_nb]) # dictionnary that contains arrays without information about the atlas labels 


            if structure==self.seed_structure:
                labels_all.append(self.seed_names)
                labels=self.seed_names
            else:

                labels_all.append(self.target_names)
                labels=self.target_names

            for i, label in enumerate(labels):
                timeseries_labels[label]=[]
                for ID_nb, ID in enumerate(self.IDs):
                    # Assign the timeseries to the label and participant ID
                    timeseries_labels[label].append(timeseries[structure]['zmean'][ID_nb][:, i].tolist())

        
        labels_all=np.concatenate(labels_all)

        if rmv_col:
            labels_df=pd.read_csv(self.atlas_labels, delimiter="\t")
            labels_df_clean=labels_df[labels_df[rmv_col]==0]
            filtered_labels = self.seed_names[np.isin(self.seed_names, labels_df_clean.iloc[:, 1].values)]
            timeseries_filtered = {label: timeseries_labels[label] for label in filtered_labels if label in timeseries_labels}




        return (timeseries_filtered, filtered_labels) if rmv_col else (timeseries, timeseries_labels, labels_all)

    def _extract_ts_atlas(self,atlas_img,img,ts_txt,redo,smoothing,structure,standardize):
        '''
            Calculate the mean time series within each atlas mask

            https://nilearn.github.io/dev/modules/generated/nilearn.maskers.NiftiLabelsMasker.html#nilearn.maskers.NiftiLabelsMasker 
        '''
        if not os.path.exists(ts_txt + '_zmean.npy') or redo==True:
            #print(ts_txt + '_zmean.npy')
            print("extracting")
            masker=NiftiLabelsMasker(labels_img=atlas_img,smoothing_fwhm=smoothing,standardize=standardize)
            ts_zmean=masker.fit_transform(img) 
            np.save(ts_txt + '_zmean.npy',ts_zmean,allow_pickle=True)
        
        else:
            print("loading")
            ts_zmean=np.load(ts_txt + '_zmean.npy',allow_pickle=True)

        return ts_zmean
    
    def correlation_df(self,seed_dict,target_dict,labels_list=None,scaling_method="zscore",partial_ts=None,groups=None,labels=None,population_info=None,tag="",save_maps=True,mean=False,smoothing_output=None,redo=False,n_jobs=1):
        '''
        Compute correlation between two seeds (seed2seed)
        Inputs
        ----------
        seed_ts: dict
            dictionnary that combined timecourse of all input seeds (see extract_data method). It could be the mean or the PC for exemple
            seed_dict=["my_seedname1":[1,-1,2,3],"my_seedname2":[-1,-4,3,4]]
        target_dict: dict
            dictionnary that combined timecourse of all input seeds (see extract_data method). It could be the mean or the PC for exemple 
            seed_dict=["my_seedname1":[1,-1,2,3],"my_seedname2":[-1,-4,3,4]]
        norm: bool
            If True, the time series will be normalized (z-scored) before computing the correlation.
        
        partial_ts: list
           List of time series to remove from the target signal (one per participant) (default = None)

        redo: boolean
            to rerun the analysis
        njobs: int
            number of jobs for parallelization
    
        Output
        ----------
        correlations: df

        '''

        results=[]

        # select info about the population
        if population_info is None:
            population_info=self.config["project_dir"]+ self.config["population_infos"]
        population_info_df=pd.read_csv(population_info,delimiter='\t')

        # Compute correlation
        output_indiv_file=[];matrix_indiv_file=[]
        
        #Create output directory
        output_indiv_dir=self.outputdir+"/1_first_level/correlation/"
        os.makedirs(output_indiv_dir, exist_ok=True)

        for ID in self.IDs:
            output_indiv_file.append(output_indiv_dir+ "/sub-" + ID + "_"+self.kind+"_df" + tag)
            matrix_indiv_file.append(output_indiv_dir+ "/sub-" + ID + "_"+self.kind+"_matrix" + tag)
            if partial_ts==None:
                partial_ts = [None] * len(self.IDs)

        # individual analysis
        if labels_list is not None and len(self.structure)==1:
            seed_names =labels_list
            target_names=labels_list
        elif len(self.structure)>1:
            #concatenate self.seed_names array and self.target_names array
            seed_names = np.concatenate((self.seed_names, self.target_names))
            target_names = np.concatenate((self.seed_names, self.target_names))
        else:
            seed_names = self.seed_names
            target_names = self.target_names
        
        if not os.path.exists(output_indiv_file[0] + ".csv") or redo==True:
            for seed_nb, seed_name in enumerate(seed_names):
                seed_name=seed_names[seed_nb]
                for target_nb, target_name in enumerate(target_names):
                    correlations = Parallel(n_jobs=n_jobs)(delayed(self._compute_connectivity)(
                        ID_nb, 
                        seed_dict[seed_name][ID_nb],
                        target_dict[target_name][ID_nb],
                        "seed2seed",
                        self.kind,
                        partial_ts[ID_nb],
                        True)
                    for ID_nb in range(len(self.IDs)))
                    
                    
                    for ID_nb, ID in enumerate(self.IDs):# Append the result as a dictionary
                        # note that the same pair is saved twice, once as seed1-seed2 and once as seed2-seed1
                        if seed_name == target_name:
                            correlations[ID_nb] = np.nan
                        results.append({
                        'IDs':ID,
                        'group': population_info_df[population_info_df["participant_id"] == ID]["group"].values[0],
                        'age': population_info_df[population_info_df["participant_id"] == ID]["age"].values[0],
                        'sex': population_info_df[population_info_df["participant_id"] == ID]["sex"].values[0],
                        "seed1": seed_name,
                        "seed2": target_name,
                        'corr': correlations[ID_nb],
                        'fcorr':np.nan,
                        'zfcorr':np.nan
                        })
                        

            # Create dataframe_______________________________________
            df = pd.DataFrame(results) 


            for ID_nb, ID in enumerate(self.IDs):# Append the result as a dictionary
                indiv_df= df[df['IDs'] == ID].copy() # Filter the dataframe for the current ID
                indiv_df['edge_sorted'] = indiv_df.apply(lambda row: tuple(sorted([row['seed1'], row['seed2']])), axis=1) # unique pair of seeds identification
                df_unique = indiv_df.drop_duplicates(subset='edge_sorted')[['edge_sorted', 'corr']].copy() # remove duplicates based on the sorted pairs, keep first occurrence
                corr = df_unique['corr'].values # Extract correlation values for this ID
                fcorr = np.full_like(corr, np.nan, dtype=np.float64) # Initialize fcorr with NaNs
                valid_mask = ~np.isnan(corr) & (np.abs(corr) < 1)
                fcorr[valid_mask] = np.arctanh(corr[valid_mask]) # Compute Fisher z-transform for valid correlations

                zcorr_zscored = np.full_like(corr, np.nan, dtype=np.float64) # Initialize zcorr_zscored with NaNs
                valid_mask = ~np.isnan(corr) & (np.abs(corr) < 1)
                zcorr_zscored[valid_mask] = zscore(corr[valid_mask])

                

                df_unique['fcorr'] = fcorr # Update the 'fcorr' column in the dataframe
                df_unique['zfcorr'] = zcorr_zscored # Update the 'z
                edge_to_zfcorr = dict(zip(df_unique['edge_sorted'], df_unique['zfcorr']))
                edge_to_fcorr = dict(zip(df_unique['edge_sorted'], df_unique['fcorr']))
                indiv_df['fcorr'] = indiv_df['edge_sorted'].map(edge_to_fcorr)
                indiv_df['zfcorr'] = indiv_df['edge_sorted'].map(edge_to_zfcorr)
                

                # Store results in dataframe
                df.loc[df['IDs'] == ID, 'fcorr'] = indiv_df['fcorr']
                df.loc[df['IDs'] == ID, 'zfcorr'] = indiv_df['zfcorr']
        
            if labels:
                if self.structure[0]=="spinalcord":
                    df['labels1'] = df.apply(lambda row: self._assign_labels1(row["seed2"], row["seed1"], self.config["labels1"]), axis=1)
                    df['labels2'] = self._assign_labels2(df, self.config["labels2"])# Add the labels2 column based on labels1
                    df['seed_level'] = df["seed1"].str.extract(r'C(\d+)').astype(float) #Extract the levels after "PAM50_C" in both seed_names and target_names
                    df['target_level'] = df["seed2"].str.extract(r'C(\d+)').astype(float)
                    df['distance_labels'] = df.apply(lambda row: "null" if row['labels1'] == "null" else abs(row['seed_level'] - row['target_level']), axis=1) # Calculate the absolute difference between the levels
                    df['level_labels'] = df.apply(lambda row: self._assign_labels1(row["seed2"], row["seed1"], self.config["level_labels"]), axis=1)
                    df = df.drop(columns=['seed_level', 'target_level']) # Drop the helper columns if needed
                    df['betwith_labels'] = df.apply(lambda row: 'intra' if row['level_labels'] != "null" and row['labels1'] != "null" 
                                else ('inter' if row['level_labels'] == "null" and row['labels1'] != "null" else "null"), axis=1)
                
                
                elif self.structure[0]=="brain":
                    df['right_left'] = df.apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["labels_RL_brain"]), axis=1)
                    df['ventro_dorsal'] =df.apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["labels3_brain"]), axis=1)
                    df['structure'] = df.apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["labels1_brain"]), axis=1)
                    df['networks'] = df.apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["network_labels"]), axis=1)
                    df['betwith_labels'] = df.apply(lambda row: 'null' if np.isinf(row['corr']) else ('intra' if row['networks'] != "null" else 'inter'), axis=1)

            # transform the df into a matrix:
            
            mat=[]
            for ID_nb, ID in enumerate(self.IDs):
                matrix=df[df["IDs"]==ID].pivot_table(
                 index="seed1", 
                 columns="seed2", 
                 values="fcorr", 
                 sort=False  )# preserve original label order
                
                
                # save individual dataframe
                mat.append(matrix.to_numpy())
                df_indiv=df[df["IDs"]==ID]
                df_indiv.to_csv(output_indiv_file[ID_nb] + ".csv", index=False)
                pd.DataFrame(mat[ID_nb]).to_csv(matrix_indiv_file[ID_nb] + ".csv", index=False, header=False)

                #save the half matrix
                df_copy = df_indiv.copy()
                df_copy["seed1"] = df_copy["seed1"].astype(str)
                df_copy["seed2"] = df_copy["seed2"].astype(str)
                df_copy['pair'] = df_copy.apply(lambda row: tuple(sorted([row["seed1"], row["seed2"]])), axis=1)
                df_upper = df_copy.drop_duplicates(subset=['IDs', 'pair'])
                df_upper = df_upper[df_upper["seed1"] != df_upper["seed2"]]
                df_upper = df_upper.drop(columns='pair')
                pd.DataFrame(df_upper).to_csv(output_indiv_file[ID_nb] + "_half.csv", index=False, header=True)

        else:
            for ID_nb, ID in enumerate(self.IDs):
                df_indiv=pd.read_csv(output_indiv_file[ID_nb] + ".csv")
                df_indiv_half=pd.read_csv(output_indiv_file[ID_nb] + "_half.csv")

       
        # group level analysis ___________________________
        #load indiv cvs files
        if groups:
            df_indiv = [pd.read_csv(file + ".csv") for file in output_indiv_file]  # Load each CSV into a DataFrame and store them in a list
            df_indiv_concat = pd.concat(df_indiv, axis=0)  # Concatenate the DataFrames horizontally
            df_indiv_half = [pd.read_csv(file + "_half.csv") for file in output_indiv_file]  # Load each CSV into a DataFrame and store them in a list
            df_indiv_concat_half = pd.concat(df_indiv_half, axis=0)  # Concatenate the DataFrames horizontally
            
            #Calculate the mean and st df over the individuals
            mean_df={}
            std_df={}
            df_group_concat={}
            df_group_concat_half={}

            for group in groups + ['ALL']:  # Add 'all' condition that will regroup all groups
                if group == 'ALL':
                    df_selected = df_indiv_concat
                    df_selected_half = df_indiv_concat_half
                else:
                    df_selected = df_indiv_concat[df_indiv_concat['group']==group]
                    df_selected_half = df_indiv_concat_half[df_indiv_concat_half['group']==group]

                df_group_concat[group] = df_selected
                df_group_concat_half[group] = df_selected_half
                
                mean_df[group] = df_selected.groupby(["seed1", "seed2"], as_index=False,sort=False)[['corr','fcorr','zfcorr']].mean() # calculate the mean, sort=False ensure no alphabeticcal reordering
                std_df[group] =  df_selected.groupby(["seed1", "seed2"], as_index=False,sort=False)[['corr','fcorr','zfcorr']].std() # calculate std


                if labels:
                    if self.structure[0]=="spinalcord":
                        mean_df[group]['labels1'] = mean_df[group].apply(lambda row: self._assign_labels1(row["seed2"], row["seed1"], self.config["labels1"]), axis=1)
                        mean_df[group]['labels2'] = self._assign_labels2(mean_df[group], self.config["labels2"])# Add the labels2 column based on labels1
                        mean_df[group]['level_labels'] = mean_df[group].apply(lambda row: self._assign_labels1(row["seed2"], row["seed1"], self.config["level_labels"]), axis=1)
                        mean_df[group]['betwith_labels'] = mean_df[group].apply(lambda row: 'intra' if row['level_labels'] != "null" and row['labels1'] != "null" 
                                    else ('inter' if row['level_labels'] == "null" and row['labels1'] != "null" else "null"), axis=1)
                        mean_df[group]['seed_level'] = mean_df[group]["seed1"].str.extract(r'PAM50_C(\d+)').astype(float) #Extract the levels after "PAM50_C" in both seed_names and target_names
                        mean_df[group]['target_level'] = mean_df[group]["seed2"].str.extract(r'PAM50_C(\d+)').astype(float)
                        mean_df[group]['distance_labels'] = mean_df[group].apply(lambda row: "null" if row['labels1'] == "null" else abs(row['seed_level'] - row['target_level']), axis=1) # Calculate the absolute difference between the levels
                        mean_df[group] = mean_df[group].drop(columns=['seed_level', 'target_level']) # Drop the helper columns if needed
                
                    elif self.structure[0]=="brain":
                        mean_df[group]['right_left'] =  mean_df[group].apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["labels_RL_brain"]), axis=1)
                        mean_df[group]['ventro_dorsal'] = mean_df[group].apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["labels3_brain"]), axis=1)
                        mean_df[group]['structure'] =  mean_df[group].apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["labels1_brain"]), axis=1)
                        mean_df[group]['networks'] =  mean_df[group].apply(lambda row: util.assign_labels3(row["seed2"], row["seed1"], self.config["network_labels"]), axis=1)

                        mean_df[group]['betwith_labels'] = mean_df[group].apply(lambda row: 'null' if np.isinf(row['corr']) else ('intra' if row['networks'] != "null" else 'inter'), axis=1)

                        
                # save group dataframe
                output_group_dir=self.outputdir+"/2_second_level/correlation/"
                os.makedirs(output_group_dir, exist_ok=True)
                
                
                if group == 'ALL':
                    group_tag=''
                    ID_nb=len(self.IDs)
                else:
                    group_tag='_'+group
                    ID_nb=len(self.config['participants_IDs' + group_tag])
                output_concat_file=output_group_dir+ "/n" + str(ID_nb) + "_"+self.kind+"_concat_df_" +group+ tag 
                output_mean_file=output_group_dir+ "/n" + str(ID_nb) + "_"+self.kind+"_mean_df_" +group+ tag + ".csv"
                output_std_file=output_group_dir+ "/n" + str(ID_nb) + "_"+self.kind+"_std_df_" +group+ tag + ".csv"
        
                if not os.path.exists(output_concat_file) or redo==True:
                    df_group_concat[group].to_csv(output_concat_file+ ".csv", index=False)
                    if group == 'ALL':
                        df_group_concat_half[group].to_csv(output_concat_file+ "_half.csv", index=False)
                   
                    mean_df[group].to_csv(output_mean_file, index=False)
                    std_df[group].to_csv(output_std_file, index=False)

                else:
                    df_group_concat[group]=pd.read_csv(output_concat_file + ".csv")
                    mean_df[group]=pd.read_csv(output_mean_file)
                    std_df[group]=pd.read_csv(output_std_file)
                
                

        
        
        return df_group_concat,df_group_concat_half, mean_df 
    


    def _compute_connectivity(self,subject_nb,seed_ts,target_ts,method,kind,partial_ts,df):


        '''
        # for l1regress we used  sklearn
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        Run the correlation analyses.
        Normalized the coef value, Fisher transformation for correlation  or not  (norm == False)
        The correlation could be classical correlations (partial==False) or partial correlations (partial==True)
        For the partial option:
        > 1. we calculated the first derivative of the signal (in each voxel of the target)
        > 2. the derivative is used as a covariate in pg.partial_corr meaning that the derivative is remove for the target but no seed signal (semi-partial correlation)
        side: "positive" , "negative" or "two-sided"
            Whether unilateral or bilateral matrice should be provide default: "two-sided"
            if "positive" : all negative values will be remplace by 0 values
        Inputs
        ----------
        subject_nb: int
            the number of the subject in the list of subjects
        seed_ts: array
            the time series of the seed (n_volume, n_voxels)
        target_ts: array
            the time series of the target (n_volume, n_voxels)
        method: str
            the method to use for the connectivity analysis, could be "seed2voxels" or "seed2seed"
        kind: str
            the kind of connectivity analysis to run, could be "corr", "cov", "l1regress", "dtw", "mi", "pcorr" or "pcorr_ts"
        partial_ts: array
            the time series of the partial covariate (n_volume, n_voxels) if kind is "pcorr_ts"
        norm: boolean
            if True the connectivity values will be normalized (e.g. Fisher transformation for correlation)
        df: boolean
            if True the connectivity values will be returned as a DataFrame, else as a numpy array

       '''


        if method=="seed2voxels":
            seed_to_voxel_connectivity = np.zeros((target_ts.shape[1], 1)) # np.zeros(number of voxels,1)
            for v in range(0,target_ts.shape[1]):
                #print(v)
                if kind=='corr': # run correlation
                    seed_to_voxel_connectivity[v] = np.corrcoef(seed_ts, target_ts[:, v])[0, 1]
                
            
                elif kind=='cov': # run covariance
                    seed_to_voxel_connectivity[v] = np.cov(seed_ts, target_ts[:, v])[0, 1]
                
                elif kind=='l1regress': # linear regression with L1 prior
                    lasso_mod   = Lasso(alpha=0.001,fit_intercept = False)
                    reg_sk      = lasso_mod.fit(seed_ts.reshape(-1, 1), target_ts[:, v])
                    seed_to_voxel_connectivity[v] = reg_sk.coef_
                
                elif kind=='dtw': # run distance time warping, add normalization to the distance path
                    #seed_to_voxel_connectivity[v] = dtw.distance(seed_ts.reshape(-1, 1), target_ts[:, v],window=60)
                    
                    seed_to_voxel_connectivity[v] = dtw.distance_fast(seed_ts.reshape(-1, 1).flatten().astype(np.float64), target_ts[:, v].astype(np.float64), window=60) # calculate the DTW between the two timeseries
                    #path = dtw.warping_path_fast(seed_ts.reshape(-1, 1).flatten(), target_ts[:, v].astype(np.float64), window=60) # Extract the optimal warping path 
                    #ix, iy = zip(*path) # provide the specific matching points from both time series
                    #seed_to_voxel_connectivity[v]=seed_to_voxel_connectivity[v]/len(ix) #Normalize the DTW distance by dividing by the length of the warping path
                    #paths=[]
                
                elif kind=='mi': # run mutual information calculation
                    seed_to_voxel_connectivity[v] =mutual_info_regression(seed_ts.reshape(-1, 1), target_ts[:, v].ravel())


                
                elif kind== 'pcorr': # run partial correlation
                    target_derivative = np.zeros((target_ts.shape[0]-1, target_ts.shape[1])) # np.zeros(number of voxels,1)
                    for v in range(0,target_ts.shape[1]-1): 
                        target_derivative[:,v] = np.diff(target_ts[:, v]) # calculate the first derivative of the signal
                        df={'seed_ts':seed_ts[:-1],'target_ts':target_ts[:-1, v],'target_ts_deriv':target_derivative[:,v]}
                        df=pd.DataFrame(df) # transform in DataFrame for pingouiun toolbox
                        seed_to_voxel_correlations[v]=pg.partial_corr(data=df, x='seed_ts', y='target_ts', y_covar='target_ts_deriv').r[0] # compute partial correlation and extract the r 
                
                


        elif method=="seed2seed":
            seed_to_voxel_connectivity=[]
            if kind=='corr':
                if df ==False:
                    seed_to_voxel_connectivity = np.corrcoef(seed_ts.T, target_ts.T)
                    np.fill_diagonal(seed_to_voxel_connectivity, np.nan)
                    
                else:
                    r=np.corrcoef(seed_ts, target_ts)[0, 1]
                    #if r >= 1:
                       #r = np.nan
                    seed_to_voxel_connectivity=r


            elif kind=='cov':
                seed_to_voxel_connectivity = np.cov(seed_ts, target_ts)[0, 1]

   
            elif kind== 'pcorr': # run partial correlation with specifyed ts
                
                if np.array_equal(seed_ts, target_ts):
                    seed_to_voxel_connectivity = np.nan # if the seed and target are the same, we cannot compute a partial correlation
                else:
                    seed_ts = np.array(seed_ts)
                    target_ts = np.array(target_ts)
                    df={'seed_ts':seed_ts,'target_ts':target_ts,'partial_ts':partial_ts}
                    df=pd.DataFrame(df) # transform in DataFrame for pingouiun toolbox
                    seed_to_voxel_connectivity =pg.partial_corr(data=df, x='seed_ts', y='target_ts', y_covar='partial_ts').r.iloc[0] # compute partial correlation and extract the r 

                                                   
                    
        return  seed_to_voxel_connectivity
   
    def _assign_labels1(self,target, seed, conditions):
        '''
        #Create the 'class1' column based on the combination of 'masks' and 'seeds'
        '''
        # Check if mask and seed are the same
        if target == seed:
            return "null"

        # Iterate through the conditions and check for patterns in mask and seed
        for class_name, patterns in conditions.items():
            if any(pattern in target for pattern in patterns) and any(pattern in seed for pattern in patterns):
                return class_name

        return "null"  # Return "None" if no conditions match

    def _assign_labels2(self,df, conditions_class2):
        '''
    Assign the class2 column based on the class1 column and the conditions provided for class2.
    
    Parameters:
    df (DataFrame): The dataframe that contains a 'class1' column.
    conditions_class2 (dict): A dictionary containing the mapping for class2 based on class1.
    
    Returns:
    DataFrame: The dataframe with an additional 'class2' column.
        '''
        # Create class2 column based on the conditions from class1
        return df['labels1'].apply(lambda x:next((key for key, values in conditions_class2.items() if x in values), "null"))
        #df['class2'] = df['class1'].apply(lambda x: next((key for key, values in conditions_class2.items() if x in values), None))
            
        """
        Computes mean within-system and between-system correlations and calculates 
        the system segregation measure for each functional system, preserving original metadata.
        
        Parameters
        ----------
        connectivity_df: pd.DataFrame
            DataFrame containing connectivity information with columns including:
            "seed1", 'corr', 'level_labels', and 'labels1'.
            
        Returns
        -------
        pd.DataFrame
            DataFrame summarizing the results with mean within-system correlations, 
            mean between-system correlations, and the segregation measure for each system.
        """

        # Calculate within-system connectivity
        within_connectivity = (connectivity_df.loc[
                (connectivity_df['betwith_labels'] == "intra")   # Assuming you meant to check a different column for "ventral"
            ]
            .groupby(['IDs', 'group', 'age', 'sex'])
            .agg(Within_Connectivity=('corr', 'mean'))
        ).reset_index()

        # Calculate between-system connectivity
        between_connectivity = (connectivity_df.loc[
                (connectivity_df['betwith_labels'] == "inter")   # Assuming you meant to check a different column for "ventral"
            ]
            .groupby(['IDs', 'group', 'age', 'sex'])
            .agg(Between_Connectivity=('corr', 'mean'))
        ).reset_index()

        # Merge both results on the specified columns
        summary = pd.merge(within_connectivity, between_connectivity, on=['IDs', 'group', 'age', 'sex'], how='outer')

        # Calculate segregation measure
        summary['Segregation Measure'] = (
            (summary['Within_Connectivity'] - summary['Between_Connectivity'])  /
            (summary['Within_Connectivity']) 
        ).fillna(0)  # Handle division by zero

        summary['Integration Measure'] = (
            (summary['Between_Connectivity']- summary['Within_Connectivity']) /
            (summary['Between_Connectivity'] +summary['Within_Connectivity']) 
        ).fillna(0)  # Handle division by zero




        return summary