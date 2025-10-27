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