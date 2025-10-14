# -*- coding: utf-8 -*-
import glob, os
import pandas as pd
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from scipy.stats import zscore
from scipy.stats import trim_mean
import brsc_utils as utils

import xml.etree.ElementTree as ET
import re

class StructuralMetrics:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, config,contrast=None, metric=None,structure="spinalcord"):
        '''
        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters
        contrast : string
            A string containing the contrast to extract could be MTR or T1w or T2s or dwi
        '''

        #1. Check the inputs and initialize the class
        if contrast not in ["MTR","T1w","T2s","DWI"]:
            raise Warning("A contrast should be provided: MTR or T1w or T2s or DWI")
        if contrast=="DWI" and metric==None:
            
            raise Warning("A metric should be provided: FA or MD or AD or RD")

        self.config = config # load config info
        self.config["project_dir"]=config["project_dir"]
        self.structure=structure
        self.contrast=contrast
        self.metric=metric
        self.IDs= config["participants_IDs_" + contrast] # list of participants to process
        print("Your are going to run the analysis here:")
        self.population_info=self.config["project_dir"] +config["population_infos"]
        
        #2. Create the output directories
        print(self.config["project_dir"], self.config["analysis_dir"][self.contrast])
        self.outputdir = self.config["project_dir"] + self.config["analysis_dir"][self.contrast]
        self.firstlevel_dir = self.outputdir+ self.config["first_level"]
        self.secondlevel_dir = self.outputdir+ self.config["second_level"]


        os.makedirs(self.outputdir, exist_ok=True) # Create directories only if they do not exist
        os.makedirs(self.firstlevel_dir, exist_ok=True) # Create directories only if they do not exist
        os.makedirs(self.secondlevel_dir, exist_ok=True) # Create directories only if they do not exist

        
        #3. Load the individual images
        print("")
        print("Start the analysis on: " + str(len(self.IDs))+ " participants")
        self.file_anat={};self.file_cord={}; self.file_gm={}; self.file_wm={}; self.file_levels={};self.file_rois={}
        for ana_space in ["indiv_space"]:#,"PAM50_space"]: #analyse could be either in individual or PAM50 space
            self.file_anat[ana_space]=[];self.file_cord[ana_space]=[]; self.file_gm[ana_space]=[]; self.file_wm[ana_space]=[]; self.file_levels[ana_space]=[]; self.file_rois[ana_space]=[]
            for ID in self.IDs:
                preproc_dir= self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
                
                self.file_rois[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][contrast + "_atlas"].format(ID))[0])
                self.file_levels[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][contrast + "_levels"].format(ID))[0])

                if contrast !="DWI":
                    self.file_anat[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][self.contrast].format(ID))[0])
                else:
                    self.file_anat[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][self.metric].format(ID))[0])


                if self.contrast =="T2s":
                    self.file_cord[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][self.contrast+"_cord"].format(ID))[0])
                    self.file_gm[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][self.contrast+"_vx_gm"].format(ID))[0])
                    self.file_wm[ana_space].append(glob.glob(preproc_dir+ self.config[ana_space][self.contrast+"_vx_wm"].format(ID))[0])
                    
                     

    def compute_csa(self,IDs=None,i_tag="cord",o_file=None,level_img=None,levels="1:8",participant_info_file=None,output_group_file=None,ana_space="indiv_space",redo_indiv=False,redo_group=False,verbose=False,n_jobs=1):
        
        '''
        Extract segmentation metrics: csa for Cross-sectional area
        https://spinalcordtoolbox.com/stable/user_section/command-line.html#sct-process-segmentation
        
        Inputs
        ----------
        IDs: list
            list of the participants to process (e.g. ["A001","A002"])
        i_tag: tag of the filename
            default: cord other options: "gm" or "wm"
        
        o_img: filename
            filename of the output (should be .csv file)
        
        level_img: filename
            The label file that specifies vertebral or segmental levels. 

        levels string
            level to process eg. 1:9
        
        participant_info: filename
            filename containing participant informations (.tsv file)
        
        redo_indiv: 
            if True the extraction will be rerun else if the timeseries were already extracted, the file containing the data will be loaded
        n_jobs: 
            Number of jobs for parallelization
             
        Returns
        ----------
          
        '''

        # 1. check the inputs info _____________________________________
        if IDs==None:
            IDs=self.IDs

        if i_tag=="cord":
            i_img=self.file_cord[ana_space]
        elif i_tag=="gm":
            i_img=self.file_gm[ana_space]
        elif i_tag=="wm":
            i_img=self.file_wm[ana_space]
        else:
            raise Warning("i_tag should one of this options: 'cord' 'gm' 'wm")

        if o_file==None:
            os.makedirs(self.firstlevel_dir + "/csa/",exist_ok=True)
            o_file= self.firstlevel_dir + "/csa/sub-{}_"+i_tag+"_csa.csv"

        if level_img==None:
            level_img=self.file_levels[ana_space]
        
        # 2. Run the analysis
        csa_files=Parallel(n_jobs=n_jobs)(delayed(self._process_csa)(ID,i_img[ID_nb],o_file.format(ID),level_img[ID_nb],levels,redo_indiv,verbose)
                                       for ID_nb, ID in enumerate(IDs))
       

       #3 . group the individual results in a group level dataframe
        if output_group_file ==None:
            os.makedirs(self.secondlevel_dir + "/csa/",exist_ok=True)
            output_group_file = self.secondlevel_dir + "/csa//n"+ str(len(IDs))+"_"+i_tag+"_csa.csv"
        print(output_group_file)
        if not os.path.exists(output_group_file) or redo_group==True:
            if participant_info_file==None:
                participant_info_file=self.population_info
            participant_info = pd.read_csv(participant_info_file,delimiter="\t")
            
            all_data = []# Create an empty list to store the individual dataframes
            vert_to_level = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'C8', 9: 'T1'}
            for ID_nb, ID in enumerate(IDs):
                file = csa_files[ID_nb]
                df = pd.read_csv(file)  # Load the individual CSV file
                participant_row = participant_info[participant_info['participant_id'] == ID]  # Extract the corresponding participant info

                if not participant_row.empty:
                    # Add the new columns to the dataframe
                    df['IDs'] = ID
                    df['groups'] = participant_row['group'].values[0]
                    df['age'] = participant_row['age'].values[0]
                    df['sex'] = participant_row['sex'].values[0]

                    # Map 'VertLevel' to 'level_labels' using the dictionary
                    df['level_labels'] = df['VertLevel'].map(vert_to_level)

                    # Reorder the columns: make 'IDs', 'group', 'age', 'sex' first
                    cols = ['IDs', 'groups', 'age', 'sex','level_labels'] + [col for col in df.columns if col not in ['IDs', 'group', 'age', 'sex','level_labels']]
                    df = df[cols]

                else:
                    print(f"Participant info not found for {ID}")

                all_data.append(df)

            # Concatenate all the individual dataframes into a single dataframe
            combined_df = pd.concat(all_data, ignore_index=True)
            # Save the combined dataframe to a new CSV file
            
            combined_df.to_csv(output_group_file, index=False)
        else:
            combined_df = pd.read_csv(output_group_file)

        
        return csa_files, combined_df


    def _process_csa(self,ID,i_img,o_file,level_img,levels,redo,verbose):
        '''
        Extracts csa
        '''
        
        if not os.path.exists(o_file) or redo==True:
            string="sct_process_segmentation -i " + i_img + " -o "+o_file+ " -vert " + levels + " -vertfile " + level_img + " -perlevel 1" 
            os.system(string)
            print("CSA was computed for sub-" + ID)
        
        #elif verbose==True:
                #print("CSA was already computed for sub-" + ID)

        return o_file

    def extract_metric_rois(self, IDs=None,input_f=None,atlas_f=None,atlas_labels=None,metric="MTR",measure="mean",sub_metrics=None,space="PAM50_space",tag="",norm=False,redo=False,verbose=1):
        '''Extract metric values in specific ROIs and put them into a dataframe
        sub_metric: list of sub_metrics for MTR their are no submetric but for DWI it could be ["FA","MD"]
        measure: can be mean (to extract the mean value within the mask) or sum (to count the number of voxels within the maskxfc              jjsaaaaaaaaa   l;;;;;;^)
        '''
        # Initialize the function
        print(metric + f' IN ROIS')
        if redo==True:
            print(f'Overwritting old files ')

        output_indiv_dir=self.firstlevel_dir + "/roi_metric/"
        if not os.path.exists(output_indiv_dir):
            os.makedirs(output_indiv_dir)
        
        if IDs==None:
            IDs=self.IDs
        
        # select atlas files
        if atlas_f is None and space=="PAM50_space":
            atlas_f=glob.glob(self.config['project_dir'] + self.config['template'][self.structure]['atlas'])[0]
            atlas_labels_f=glob.glob(self.config["project_dir"] + self.config['template'][self.structure]['atlas'].split("order")[0] + "labels.txt")[0]
            atlas_labels=np.genfromtxt(atlas_labels_f, usecols=0,skip_header=1, dtype="S", delimiter="\t", encoding=None)
            atlas_labels = np.array([label.decode("utf-8") for label in atlas_labels])
 
            
        elif atlas_f is None and space=="indiv_space":
            atlas_f=[]
            for ID in self.IDs:
                preproc_dir=self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
                atlas_f.append(glob.glob(preproc_dir + self.config["indiv_space"][metric + "_atlas"].format(ID))[0])
        
            if atlas_labels is None:
                raise Warning("Atlas labels should be provided, you can for instance provide a list of numbers from 1 to n")

        # Select the input data files
        masked_data_list = [];data_files={}
        if input_f is None:
            if sub_metrics:
                for sub_metric in sub_metrics:
                    data_files[sub_metric]={}
                    for ID in self.IDs:
                        preproc_dir=self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
                        data_files[metric][ID]=glob.glob(preproc_dir + self.config[space][sub_metric].format(ID))[0]

            else:
                data_files[metric]={}
                for ID in self.IDs:
                    preproc_dir=self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
                    data_files[metric][ID]=glob.glob(preproc_dir + self.config[space][metric].format(ID))[0]       
        else:
            data_files[metric]={}
            for ID_nb, ID in enumerate(self.IDs):
                data_files[metric][ID]=input_f[ID_nb]

        # Loop over the IDs and extract the metric values and put them into a dataframe

        all_data=[]
        all_metrics=[metric] if sub_metrics==None else sub_metrics
        metadata = pd.read_csv(self.population_info, delimiter='\t')

        for metric1 in all_metrics:
            for ID_nb, ID in enumerate(IDs):
                if not os.path.exists(output_indiv_dir+"/sub-"+ID+"_"+metric1+tag +".csv") or redo:  
                    data_file=data_files[metric1][ID]
                    data_img = nib.load(data_file) # load data from the image file
                    data_data = data_img.get_fdata() # put the data into a numpy array
                    
                    IDs_region=[]
                    IDs_sub = [] ; rois = [] ; groups = [] ; ages=[]; sex=[]
                    for label_nb, label in enumerate(atlas_labels):
                        # Get participant metadata
                        IDs_sub.append(ID)
                        groups.append(metadata [metadata ["participant_id"] == ID]["group"].values[0])
                        ages.append(metadata [metadata ["participant_id"] == ID]["age"].values[0])
                        sex.append(metadata [metadata ["participant_id"] == ID]["sex"].values[0])
                        rois.append(label)
            
                        # extract the metric values in the ROIs
                        if space=="PAM50_space":
                            atlas_img = nib.load(atlas_f)
                        
                        elif space=="indiv_space":
                            atlas_img =  nib.load(atlas_f[ID_nb])
                        
                        atlas_data= atlas_img.get_fdata() # put the data into a numpy array
                        mask=atlas_data==label_nb+1 # sine the loop starts at 0 and the atlas labels start at 1
                        

                        if np.any(mask):
                            values=data_data[mask] # extract the data_data values where atlas_data==label_nb+1
                        else:
                            values=np.nan # else put nan
                        
                        if measure=="mean":
                            IDs_region.append(np.nanmean(values)) # calculate the mean value of the data_data in the mask
                        
                        if measure=="median":
                            IDs_region.append(np.nanmedian(values)) # calculate the median value of the data_data in the mask
                   
                        if measure=="count":
                            nonzero_voxel_count = np.count_nonzero(data_data[mask]) # count the number of non-zero voxels in the mask
                            IDs_region.append(nonzero_voxel_count)
                            

                    ID_values = np.array(IDs_region)

                    if norm and measure=="mean":
                        ID_values = zscore(ID_values, nan_policy='omit')

                    #create the individual dataframe
                    colnames = ["IDs","age","sex","groups","rois",metric1]
                    df_metric = pd.DataFrame(list(zip(IDs_sub, ages,sex,groups, rois, ID_values)), columns=colnames)


                    if self.config["labels1"] and self.structure=="spinalcord":
                        df_metric['ventro_dorsal'] = df_metric.apply(lambda row: utils.assign_labels1(row['rois'], self.config["labels_VD"]), axis=1)
                        # Add the class2 column based on class1
                        df_metric['right_left'] = df_metric.apply(lambda row: utils.assign_labels1(row['rois'], self.config["labels_RL"]), axis=1)

                        df_metric['levels'] = df_metric.apply(lambda row: utils.assign_labels1(row['rois'], self.config["level_labels"]), axis=1)
                   

                    #save as csv
                    df_metric.to_csv(output_indiv_dir + f"/sub-{ID}_{metric1}{tag}.csv",index=False)
                
                else:
                    df_metric=pd.read_csv(output_indiv_dir + f"/sub-{ID}_{metric1}{tag}.csv")

                all_data.append(df_metric)

            # Concatenate all the individual dataframes into a single dataframe
            os.makedirs(self.secondlevel_dir+"/roi_metric/", exist_ok=True)
            output_all_f=self.secondlevel_dir + "/roi_metric/n" + str(len(IDs))+ "_" + metric1 + tag + ".csv"
            all_df = pd.concat(all_data, ignore_index=True)

            print(output_all_f)
            if not os.path.exists(output_all_f) or redo:
                # Save the combined dataframe to a new CSV file
                all_df.to_csv(output_all_f, index=False)
              
           
        return all_df
    

