# -*- coding: utf-8 -*-
import glob, os
import pandas as pd
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from scipy.stats import zscore, iqr
import brsc_utils as util




class Matrix:
    def __init__(self, config,IDs=None,output_dir=None,metrics=None,structure="spinalcord",space="PAM50_space"):
        """
        Initialize the CovarianceMatrix class with the config file
        
        Args:   
        - config: str, path to the configuration file.
        - structure: str, structure to be used for the covariance matrix. either "spinalcord" or "brain".
        """
        self.config=config
        self.space=space 
        self.config = config
        if IDs is None:
            self.IDs = self.config["participants_IDs_ALL"]
        else:
            self.IDs = IDs
        
        self.structure=structure
        self.population_info=self.config["project_dir"] +config["population_infos"]
        self.metadata=pd.read_csv(self.population_info, delimiter='\t')

        self.output_dir=output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)     


    def compute_similarity_matrix(self,data_df=None,column_labels=None,atlas_labels=None, roi_col="rois",tag="",structure="spinalcord",scaling_method=None,redo=False):
        """
        Compute the similarity matrix using the atlas and the individual data.
        The similarity matrix is computed for each individual and stored in a dictionary.
        Args:
        - data_df: pd.DataFrame, DataFrame containing the individual data.
        - column_labels: list, list of column labels to extract from the DataFrame.
        - roi_col: str, column name to use as the seed for similarity computation. default is "rois".
        - tag_name: str, tag name to append to the output file name.

        Returns:
        - sim_matrix: np.ndarray, similarity matrix of the masked data.
        """
        # Initialize an empty list to store the masked data
        if data_df is None:
            raise ValueError("data_df must be provided")
        
        if column_labels is None:
            column_labels=data_df.columns[0:]
            print("Extraction is going to be done for the following features : " )
            print(data_df.columns)

        # Create output directory
        sim_dir = os.path.join(self.output_dir, "1_first_level/sim_matrix/")
        os.makedirs(sim_dir, exist_ok=True) 

        all_sim_matrix=[];output_f=[];all_sim_dfs=[]
        for ID_nb, ID in enumerate(self.IDs):
            output_f.append(sim_dir + f"/sub-{ID}_sim_matrix{tag}.csv")

            if not os.path.exists(output_f[ID_nb]) or redo:
                sub_df = data_df[data_df['IDs'] == ID]
                roi_feature_matrix = []  # (n_rois, n_contrasts) at the end
                roi_zscored=[]
                roi_labels=[]

                for feature in column_labels:
                    roi_values=[]
                    for label_nb, label in enumerate(data_df[roi_col].drop_duplicates().values):
                        val = sub_df[sub_df[roi_col] == label][feature].values # get the metric value for this ROI and feature
                        if len(val) > 0:
                            roi_values.append(val[0])
                        else:
                            roi_values.append(np.nan)
                        roi_labels.append(label)  # Store the label for later use

                    
                    # Apply z-scoring to the roi_values

                    if scaling_method=="zscore":
                        roi_zscored.append(zscore(roi_values, nan_policy='omit'))
                    elif scaling_method=="robust_sigmoid":
                        roi_zscored.append(util.robust_sigmoid(roi_values))
                    elif scaling_method=="mad":
                        roi_zscored.append(self._MAD(roi_values))
                    elif scaling_method is None:
                        roi_zscored.append(roi_values)
                    else:
                        raise ValueError(f"Unknown scaling method: {scaling_method}. Use 'mad', 'zscore', or 'robust_sigmoid'.")
                    
                roi_feature_matrix = np.array(roi_zscored)
                
                # Similarity matrix computation (rois x rois)
                indiv_sim_matrix = np.corrcoef(roi_feature_matrix, rowvar=False)  # Compute the correlation matrix
                np.fill_diagonal(indiv_sim_matrix, np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    indiv_sim_matrix = np.arctanh(indiv_sim_matrix)# Apply Fisher transform to off-diagonal values only 
                
                all_sim_matrix.append(indiv_sim_matrix)
                np.savetxt(output_f[ID_nb], all_sim_matrix[ID_nb], delimiter=",") #save output file
            else:
                all_sim_matrix.append(np.genfromtxt(output_f[ID_nb], delimiter=",", filling_values=np.nan))
            
            # --- Built output dataframe ---
            
            sim_df=self._matrix2dataframe(all_sim_matrix[ID_nb],ID,self.metadata,atlas_labels,structure,variable="sim")
            all_sim_dfs.append(sim_df.reset_index(drop=True))  # Reset index for alignment
        
        final_df = pd.concat(all_sim_dfs, ignore_index=True)
        # Save the final DataFrame to a CSV file
        print(os.path.join(sim_dir, f"sim_matrix_df{tag}.csv"))
        final_df.to_csv(os.path.join(sim_dir, f"sim_matrix_df{tag}.csv"), index=False)



        # Group level similarity matrix
        mean_sim_matrix = np.nanmean(np.stack(all_sim_matrix), axis=0)

        return all_sim_matrix, mean_sim_matrix,final_df
    
    def _matrix2dataframe(self,matrix,ID,metadata,atlas_labels,structure,variable="var"):
        """
        Convert a matrix to a DataFrame with additional information from the participant_info DataFrame.
        Args:
        - matrix: np.ndarray, the matrix to convert.
        - ID: str, the participant ID.
        - info: pd.DataFrame, the participant_info DataFrame containing additional information.
        - atlas_label: list, the labels of the atlas regions.
        - variable: str, the name of the variable to be added to the DataFrame.
        Returns:
        - df: pd.DataFrame, the resulting DataFrame with additional information.
        """
        #Get individual metadata
        if metadata is not None:
            participant_row = metadata[metadata['participant_id'] == ID]
    
            if participant_row.empty:
                print(f"Warning: No participant info found for {ID}")
                age = None
                sex = None
                group = None
            else:
                age = participant_row["age"].values[0]
                sex = participant_row["sex"].values[0]
                group = participant_row["group"].values[0]
        
        
        # Create a DataFrame from the matrix
        rows = []
        n_regions = len(atlas_labels)
        region_labels = atlas_labels

        for i in range(n_regions):
            for j in range(i+1, n_regions):
                label_i = region_labels[i]
                label_j = region_labels[j]
            
                if metadata is not None:
                    rows.append({
                        "IDs": ID,
                        "age": age,
                        "sex": sex,
                        "group": group,
                        "seed1": label_i,
                        "seed2": label_j,
                        variable: matrix[i, j]
                    })
                
                else:
                    
                    rows.append({
                        "seed1": label_i,
                        "seed2": label_j,
                        variable: matrix[i, j]
                    })
            
        df = pd.DataFrame(rows) 
        if structure == "spinalcord":
            df['labels1'] = df.apply(lambda row: util.assign_labels3(row['seed1'], row['seed2'], self.config["labels1"]), axis=1)
            df['labels2'] = util.assign_labels2(df, self.config["labels2"])# Add the labels2 column based on labels1
            df['level_labels'] = df.apply(lambda row: util.assign_labels3(row['seed1'], row['seed2'], self.config["level_labels"]), axis=1)
            df['betwith_labels'] = df.apply(lambda row: 'intra' if row['level_labels'] != "null" and row['labels1'] != "null" 
                                        else ('inter' if row['level_labels'] == "null" and row['labels1'] != "null" else "null"), axis=1)
        elif structure == "brain":
            df['labels1'] = df.apply(lambda row: util.assign_labels3(row['seed1'], row['seed2'], self.config["labels1_brain"]), axis=1)
            df['labels2'] = util.assign_labels2(df, self.config["labels3_brain"])# Add the labels2 column based on labels1
            df['network_labels'] = df.apply(lambda row: util.assign_labels3(row['seed1'], row['seed2'], self.config["network_labels"]), axis=1)
            df['betwith_labels'] = df.apply(lambda row: 'intra' if row['network_labels'] != "null" 
                                        else 'inter' if row['network_labels'] == "null" else "null", axis=1)



            
        return df
    
    def _MAD(self,x):
        """ We applied a non-parametric equivalent of the Z-score using median and MAD (MAD, median absolute deviation; Md, median).
            see F. Váša et al., Rapid processing [...]. Hum. Brain Mapp. 43, 1749–1765 (2022).
        """
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))
                      
        if mad == 0 or np.isnan(mad):
            return np.zeros_like(x)  # Avoid division by zero
        else:
            return (x - median) / mad  

