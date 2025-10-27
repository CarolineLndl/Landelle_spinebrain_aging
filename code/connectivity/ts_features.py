import pandas as pd
import nibabel as nib
import numpy as np
import os, glob
from joblib import Parallel, delayed
import antropy as ant
from statsmodels.tsa.stattools import kpss
import warnings
from scipy.stats import zscore, iqr
import pycatch22
import brsc_utils as util

class FeatureSets:
    '''
    The FeatureSets class is used to compute time-series analyses.
    This class include different time-series metrics, such as:
        - coefficient of variation (CV) 
        - Sample entropy (SampEN)
        - Autocorrelation (AutoCorr)
        - Stationarity with KPSS (KPSS)
    
    nb: for Power spectral properties see alff.py


    For reference, see:
    Ben D. Fulcher, Nick S. Jones, hctsa: A Computational Framework for Automated Time-Series Phenotyping Using Massive Feature Extraction,
    Cell Systems, Volume 5, Issue 5, 2017,
    https://doi.org/10.1016/j.cels.2017.10.001.

    For toolbox in matlab, see:
    https://github.com/benfulcher/hctsa/tree/main/

    Attributes
    ----------
    config : dict
        Contains information regarding IDjects, runs, rois, etc.
    '''

    def __init__(self, config,IDs,structure='spinalcord',verbose=1):  
        '''
        Parameters
        ----------
        config : dict
            Contains information regarding IDjects, runs, rois, etc.
        IDs : list
            List of IDs to be analyzed
        structure : str
            Structure to be analyzed (default is 'spinalcord')
        verbose : int
            Verbosity level (default is 1)
        '''

        self.config = config
        self.IDs=IDs
        self.structure=structure
        self.population_info=self.config["project_dir"] +config["population_infos"]
        self.metadata=pd.read_csv(self.population_info, delimiter='\t')
        
        # define output directory and create if it does not exist
        self.output_dir = config["project_dir"] + config["ts_features"]["analysis_dir"][structure] + '/' 
        self.firstlevel_dir = self.output_dir + "1_first_level/"
        self.secondlevel_dir = self.output_dir + "2_second_level/"
        
        #create the output directories if they do not exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.firstlevel_dir, exist_ok=True)
        os.makedirs(self.secondlevel_dir, exist_ok=True)

        # input files that contain the time-series data
        self.input_files=[]
        self.ts_data=[]

        
        for ID_nb, ID in enumerate(self.IDs):
            self.input_files.append(glob.glob(self.config["project_dir"] + self.config["ts_features"]["template_space"][structure]["func"].format(ID))[0]) # Get the first file that matches the pattern
            
        if verbose ==1:
            print(f'Feature analyses output directory: {self.output_dir}')
            print("")
            print(f'Analyzing {len(self.IDs)} participant(s)')

    def compute_pychatch24(self,ts_data=None,seed_labels=False,add_features=None,scaling_method="zscore",redo=False,verbose=1,n_jobs=1):
        """Computes time-series features for each voxel
        ts_data : str
            Array that contains the time-series data for each voxel or seed and each ID. (ID_nb, volumes, voxels/seeds)
            If None, the loading condition is not implemented yet
        seed_labels : bool
            If True, the output will include seed labels (default is False).
        add_features:
            if you want to add an additional feature it sould have the format: {"feature_name":feature_values}, with feature values shape: n_participants, n_seeds
        scaling_method : str
            Method to scale the features. Options are "zscore" or "robust_sigmoid" (default is "zscore").
            "zscore" applies z-score normalization to each feature.
            "robust_sigmoid" applies a robust sigmoid transformation to each feature.
        redo : bool
            Set to True to overwrite existing files
        verbose : int
            Verbosity level (default is 1)
        n_jobs : int
            Number of parallel jobs to run (default is 1)
        Outputs
        ----------
     """
    
        if ts_data is None:
            raise ValueError("ts_data should be provided as an array-like structure (ID_nb, volumes, voxels/seeds)")

        os.makedirs(self.firstlevel_dir + "/catch24/", exist_ok=True)

        # ouput file paths
        ts_f = [f"{self.firstlevel_dir}/catch24/sub_{ID}_{self.structure}_catch24.csv"
                for ID in self.IDs]
        
        # Compute the features for each ID
        results = []
        z_tag="_z" if scaling_method=="zscore" else "_sig"
        for ID_nb, ID in enumerate(self.IDs):
            results_indiv=[]
            if not os.path.exists(ts_f[ID_nb]) or redo==True:
                for seed_nb, seed_name in enumerate(seed_labels):
                    
                    results_indiv.append(pycatch22.catch22_all(ts_data[ID_nb][:,seed_nb],short_names=True,catch24 = True))

                
                # Build the DataFrame
                feature_names=results_indiv[0]["short_names"] # use short_names for the feature names
                feature_matrix = pd.DataFrame([r['values'] for r in results_indiv], columns=feature_names) # combine results into a DataFrame

                if add_features:
                    for new_feature in add_features:
                        feature_matrix[new_feature]=add_features[new_feature][ID_nb]
                    
                        feature_names.append(new_feature)
                
                # apply normalization for each feature
                if scaling_method=="zscore":
                    feature_matrix_z = feature_matrix.apply(zscore, axis=0, nan_policy='omit') # Apply z-score normalization to each column, ignoring NaN values
                    
                elif scaling_method=="robust_sigmoid":
                    feature_matrix_z = feature_matrix.apply(util.robust_sigmoid, axis=0)

                # Create the columns for the DataFrame
                IDs = [] ;  groups = [] ; ages=[]; sex=[]
                ID_meta = self.metadata[self.metadata["participant_id"] == ID]
                ID_meta = ID_meta.iloc[0]  # Grab the first (and should be only) match
            
                # Create list entries per ROI
                n_rois = len(seed_labels)
                IDs = [ID] * n_rois
                groups = [ID_meta["group"]] * n_rois
                ages = [ID_meta["age"]] * n_rois
                sex = [ID_meta["sex"]] * n_rois

                #Create the DataFrame
                df_indiv = pd.DataFrame({
                "IDs": IDs,
                "age": ages,
                "sex": sex,
                "groups": groups,
                "rois": seed_labels})
                #Add additional columns for the metrics
                if self.config["labels1"] and self.structure=="spinalcord":
                    df_indiv['ventro_dorsal'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels_VD"]), axis=1)
                    # Add the class2 column based on class1
                    df_indiv['right_left'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels_RL"]), axis=1)
                    df_indiv['levels'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["level_labels"]), axis=1)

                elif self.config["labels1_brain"] and self.structure=="brain":
                    df_indiv['right_left'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels_RL_brain"]), axis=1)
                    df_indiv['ventro_dorsal'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels3_brain"]), axis=1)
                    df_indiv['structure'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["labels1_brain"]), axis=1)
                    df_indiv['networks'] = df_indiv.apply(lambda row: util.assign_labels1(row['rois'], self.config["network_labels"]), axis=1)


                df_indiv_z=df_indiv.copy()  # Create a copy of the DataFrame for z-scored features
                for i, col in enumerate(feature_names):
                    #print(feature_matrix.columns[i])
                    df_indiv[col] = feature_matrix.iloc[:, i]  # Use iloc to access the column by index
                    df_indiv_z[col] = feature_matrix_z.iloc[:, i]  # Use iloc to access the column by index

                # save the DataFrame to a file
                df_indiv.to_csv(ts_f[ID_nb], index=False) #save the results to a file
                df_indiv_z.to_csv(ts_f[ID_nb].replace(".csv", z_tag + ".csv"), index=False) #save the z-scored results to a file
            else:
                df_indiv = pd.read_csv(ts_f[ID_nb])
            df_indiv_z = pd.read_csv(ts_f[ID_nb].replace(".csv", z_tag + ".csv"))

        
        # concatenate all individual DataFrames into a single DataFrame
        df = pd.concat([pd.read_csv(f) for f in ts_f], ignore_index=True)
        df_z= pd.concat([pd.read_csv(f.replace(".csv", z_tag + ".csv")) for f in ts_f], ignore_index=True)

        n_IDs=len(self.IDs)
        df_f =f"{self.secondlevel_dir}/catch24/n{str(n_IDs)}_{self.structure}_catch24.csv"
        df_z_f =f"{self.secondlevel_dir}/catch24/n{str(n_IDs)}_{self.structure}_catch24{z_tag}.csv"
        
        if not os.path.exists(df_f) or redo==True:
            os.makedirs(self.secondlevel_dir + "/catch24/", exist_ok=True)
            df.to_csv(df_f, index=False)
            df_z.to_csv(df_z_f, index=False)

       
        return df,df_z
    

   