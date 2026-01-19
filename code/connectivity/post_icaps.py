
# -*- coding: utf-8 -*-
import glob, os, json
import numpy as np
from scipy.io import loadmat
import pandas as pd
import mat73
import nibabel as nib
import scipy

#statistics
from statistics import mean, stdev
from scipy.ndimage import center_of_mass,label,find_objects

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

class post_icaps:
    '''
    To extract relevant information about the icaps analyses
    
    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, config_file,analysis='icaps',structure='spinalcord',verbose=1):
        '''
        Parameters
        ----------
        config_file : str
            path to the config file
        analysis : str
            name of the analysis (default = 'icaps')
        structure : str
            name of the structure (default = 'spinalcord')
        verbose : int
            verbosity level (default = 1)
        '''
        self.verbose=verbose
        # check if the config file exists and load it
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file) as config_f:
            self.config = json.load(config_f)

        
        # Initialize the class with the config file
        self.config_file=config_file
        self.analysis=analysis
        self.structure=structure
        self.n_k=self.config[analysis]['n_k'][self.structure][0] # number of components
        
        print("Your are going to run the analysis here:")
        self.outputdir= self.config["project_dir"] +self.config[analysis]["analysis_dir"][self.structure]
        print(self.outputdir)

        #load participant name info
        data_f=glob.glob(self.config['project_dir'] + self.config[analysis]['analysis_dir'][self.structure] + "/param.mat")[0]
        data_dict = mat73.loadmat(data_f)
        IDs_raw=data_dict['param']['Subjects']
        self.IDs = [item.split('/')[0].split('-')[1] for item in IDs_raw] # array with ID info
        print("number of participants: " + str(len(self.IDs)))


        #load output from icaps framework analyses
        data_f=glob.glob(self.config['project_dir'] + self.config[analysis]['analysis_dir'][self.structure] + "/TCs*/tempChar.mat")[0]
        self.data = loadmat(data_f)['tempChar'][0, 0]

        # load icap image
        self.img_f=glob.glob(self.config['project_dir'] + self.config[analysis]['analysis_dir'][self.structure] + "/iCAPs_z.nii")[0]
        self.atlas_f=glob.glob(self.config['project_dir'] + self.config["templates"]['spinalcord']["spinal_levels"])[0]
        atlas_data=nib.load(self.atlas_f).get_fdata()
        if len(atlas_data.shape) < 4:
            raise ValueError(f"Expected 4D atlas data, but got {len(atlas_data.shape)}D data with shape {atlas_data.shape}.")

    
    def thresholding_masking(self,IDs=None,i_img=None,labels_f=None, lthr=1.5):
        
        '''
        IDs: array
            name of the different participant to analyse, ["A001", "A002","A003"]
        
        i_img: filename
            nfilenameame of the 4D image containing the different icaps components
        
        labels:
            You can create a labels file: iCAPs_Kn_labels.txt (delimiter > \t) with firt colunm k value and second the label that you want to attribute e.g: 
            1	Audio	
            2	Cereb

        lthr: float
            value of the lower threshold (default =1.5)
        '''
        if IDs==None:
            IDs=self.IDs

        if i_img==None:
            i_img=glob.glob(self.config["project_dir"] + self.config[self.analysis]["analysis_dir"][self.structure] + "/iCAPs_z.nii")[0]

        if labels_f==None:
            label_f=glob.glob(self.config['project_dir'] + self.config[self.analysis]['analysis_dir'][self.structure] + "/iCAPs_K"+str(self.n_k)+"_labels.txt")[0]
        label_df=pd.read_csv(label_f,delimiter="\t",header=None,usecols=[1])
        # Create output directory
        output_dir=glob.glob(self.config["project_dir"] + self.config[self.analysis]["analysis_dir"][self.structure] + "/")[0] + "comp_zscored/"

        if not os.path.exists(output_dir):
            print(output_dir)
            os.mkdir(output_dir)

        # Split 4D image into 3D images
        icaps_3d_basename=output_dir + "iCAPs_z"
        if not os.path.exists(icaps_3d_basename + self.structure + "_icap_" + label_df.loc[0,1] + ".nii.gz"):
            string_split='fslsplit ' + i_img + " " + icaps_3d_basename + " -t"
            os.system(string_split)

        # Threshold and mask 3D images
        for k_level in range(0,self.n_k):
            tag_output=self.structure + "_icap_" + label_df.loc[k_level,1]
            if k_level<10:
                default_f=glob.glob(output_dir + '/*000' + str(k_level) +".nii.gz")[0]
            else:
                default_f=glob.glob(output_dir + '/*00' + str(k_level) +".nii.gz")[0]

            icap_3d_file=output_dir + '/' + tag_output +".nii.gz"

            if not os.path.exists(icap_3d_file):
                os.rename(default_f,icap_3d_file)

            bin_file= output_dir +tag_output+ "_bin.nii.gz"
            
            if not os.path.exists(bin_file):
                string_bin='fslmaths ' +icap_3d_file+ ' -thr '+str(lthr)+' -bin ' + bin_file
                os.system(string_bin)
            


        
    def extract_metrics(self,IDs=None,data=None,info_pop_f=None,labels=None):
        '''
        data: bool
            matlab matrix converted with loadmap containing metric informations data = loadmat(filename.mat)
        
        IDs: array
            name of the different participant to analyse, ["A001", "A002","A003"]
        
        info_pop_f: filename
            the csv file should contain information about the population, with columns like: participant_id	dataset	group

        labels:
            You can create a labels file: iCAPs_Kn_labels.txt (delimiter > \t) with firt colunm k value and second the label that you want to attribute e.g: 
            1	Audio	
            2	Cereb
            ..
            n   DMN
        '''
        if data==None:
            data=self.data
        
        if IDs==None:
            IDs=self.IDs

        # Read file conaining information about the population:
        if  info_pop_f==None:
            info_pop_f=glob.glob(self.config["project_dir"] + self.config["population_infos"])[0]
        info_pop_df=pd.read_csv(info_pop_f, sep='\t')
        
        # Collect all data into a list of dictionaries
        data_1d_combined = [] ; data_2d_combined = []; data_3d_combined = []

        # Iterate over participants
        
        for idx, participant_id in enumerate(IDs):
            # Get participant information
            group = info_pop_df.loc[info_pop_df['participant_id'] == participant_id, 'group'].values[0]
            age = info_pop_df.loc[info_pop_df['participant_id'] == participant_id, 'age'].values[0]
            sex = info_pop_df.loc[info_pop_df['participant_id'] == participant_id, 'sex'].values[0]

            # Create a dictionary to store data for this participant
            participant_info = {
                'IDs': participant_id,
                'group': group,
                'age': age,
                'sex': sex
            }
            k_n=self.config['n_k'][self.structure][0]

            # Create a dict that include iCAPs information (data_2d)
            data_2d = {key: [value] * k_n for key, value in participant_info.items()}  # Repeat each scalar value k_n times
            data_2d['iCAPs'] = list(range(1, k_n + 1))  # Add iCAPS from 1 to k_n

            if labels:
                labels_info=pd.read_csv(glob.glob(self.config['project_dir'] + self.config['analysis_dir'][self.structure] + "/K_" + str(self.config['n_k'][self.structure][0]) +"*/iCAPs_K"+str(k_n)+"_labels.txt")[0],delimiter="\t",header=None)# read label file
                data_2d['iCAPs_labels'] = labels_info.iloc[:, 1]# copy label info in the df
            

            # Add metrics for this participant
            for var in data.dtype.names:
                
                if data[var][0, 0].shape[0] == 1 and data[var][0, 0].shape[1] == len(IDs):
                    data_1d=participant_info
                    if not isinstance(data[var][0][0][0][0], np.ndarray):
                        metric_1d = data[var][0, 0]
                        data_1d[var] = metric_1d[0, idx]  # Add the metric for the current variable

                if data[var][0, 0].shape[0] == k_n and data[var][0, 0].shape[1] == len(IDs):
                    if not isinstance(data[var][0][0][0][0], np.ndarray):
                        metric_2d = data[var][0, 0]
                        for k in range(0,k_n):
                            data_2d[var] = metric_2d[:, idx]  # Add the metric for the current variable
                
                if data[var][0, 0].shape[0] == k_n and data[var][0, 0].shape[1] == k_n:

                    data_3d=participant_info
                    data_3d[var] = data[var][0, 0][:, :, idx]  # Add the metric for the current variable


            # Append the combined data for this participant
            data_1d_combined.append(data_1d)
            data_2d_combined.append(data_2d)
            data_3d_combined.append(data_3d)
        
        # Create a DataFrame from the combined data
        df_1d = pd.DataFrame(data_1d_combined)

        # Converting to DataFrame
        dfs = [pd.DataFrame(participant) for participant in data_2d_combined]
        df_2d  = pd.concat(dfs, ignore_index=True)



        # Return the final DataFrame
        return df_1d,  df_2d, data_3d_combined


    def icaps_matrices(self,matrices_list=None,labels=None):
        '''
        
        '''
        if labels:
            labels_info=pd.read_csv(glob.glob(self.config['project_dir'] + self.config['analysis_dir'][self.structure] + "/K_" + str(self.config['n_k'][self.structure][0]) +"*/iCAPs_K"+str(self.config['n_k'][self.structure][0])+"_labels.txt")[0],delimiter="\t",header=None)# read label file

        #>>>>>>>>>>>>>>>>>>>>>>>> Population level <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        combined_data = []
        for participant in matrices_list:
            participant_data = {key: participant[key] for key in participant.keys()}
            combined_data.append(participant_data)
        
        # concatenate matrices from the same metric (e.g: 'coupling_counts')
        indiv_matrices = {}
        for participant in combined_data:
            for metric, value in participant.items():
                if isinstance(value, np.ndarray) and value.ndim == 2: # Check if the field contains a matrix (i.e., a 2D array)
                    if metric not in indiv_matrices:
                        indiv_matrices[metric] = []
            
                    indiv_matrices[metric].append(value)

        # Calculate the mean matrix over the individuals for each metric
        mean_matrix = {}
        for metric, matrices in indiv_matrices.items():
            matrices_array = np.array(matrices)
            mean_matrix[metric] = np.nanmean(matrices_array, axis=0) # Calculate the mean across the IDs 
    
        individual_means = []

        # Calculate row-wise means for each individual and store in DataFrame
        for participant in combined_data:
            participant_data = {
                "IDs": participant.get("IDs"),
                "age": participant.get("age"),
                "group": participant.get("group")}


            for metric, value in participant.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    row_means = np.nanmean(value, axis=1)
                    participant_data[metric] = row_means

            num_labels = self.config['n_k'][self.structure][0]
            for row_index in range(num_labels):
                individual_row = {
                    "IDs": participant_data["IDs"],
                    "age": participant_data["age"],
                    "group": participant_data["group"],
                    "iCAPs": row_index,
                    "iCAPs_labels":labels_info.iloc[row_index, 1]
                }
                # Add metrics for the current row
                for metric, values in participant_data.items():
                    if isinstance(values, np.ndarray):
                        individual_row[metric] = values[row_index]

                # Append to the list
                individual_means.append(individual_row)
            
                
        df_matrix_pop = pd.DataFrame(individual_means)
        
    #>>>>>>>>>>>>>>>>>>>>>>>> Sub groups level <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Separate data into subgroups
        subgrouped_data = {}
        for participant in combined_data:
            group = participant.get('group')  # Assuming 'group' is the field to define subgroups
            if group not in subgrouped_data:
                subgrouped_data[group] = []  # Initialize a list for each subgroup
            subgrouped_data[group].append(participant)

        
        mean_matrices_by_group = {}
        for group, participants in subgrouped_data.items():
            indiv_matrices = {}
            for participant in participants:
                for metric, value in participant.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:  # Check if the field contains a matrix (2D array)
                        if metric not in indiv_matrices:
                            indiv_matrices[metric] = []  # Initialize list if not already present
                        indiv_matrices[metric].append(value)

        
            mean_matrix = {}
            for metric, matrices in indiv_matrices.items():
                matrices_array = np.array(matrices)  # Convert list to a 3D array
                mean_matrix[metric] = np.nanmean(matrices_array, axis=0)


            mean_matrices_by_group[group] = mean_matrix

        return mean_matrices_by_group, mean_matrix, df_matrix_pop
    
    def spatial_similarity(self, k1=None, mask=None, thr=2,similarity_method='Dice', sorting_method='rostrocaudal', return_mean=False, save_results=False,save_figure=False, verbose=True):
        '''
        Compares spatial similarity for different sets of components.
        To obtain a similarity matrix for a particular K per condition
 
        Inputs
        ----------
        k1 : int
            K values of interest (default = None) 
        mask: int
            Mask to apply (default = self.config["project_dir"] + self.config["templates"]["spinalcord"]["cord"])
        thr: int
            Threshold to apply (default = 2)
        similarity_method : str
            Method to compute similarity (default = 'Dice')
                'Dice' to compute Dice coefficients (2*|intersection| / nb_el1 + nb_el2)
                'Euclidean distance' to compute the distance between the centers of mass
                'Cosine' to compute cosine similarity 
        sorting_method : str
            Method used to sort maps (default = 'rostrocaudal')
        return_mean : boolean
            Set to True to return the mean diagonal similarity (default = False)
        save_results : boolean
            Results are saved as npy or txt if set to True (Default = False)
        verbose : bool
            If True, print progress for each K (default=True)
        '''

        # Check if k values are provided & choose method accordingly
        if k1 == None : 
            raise(Exception(f' k1 needs to be specified!'))
        
        output_fname=self.outputdir +  'iCAPs_k' + str(k1) + '_vs_atlas_mean_0_'
        output_fig=self.outputdir +  '/figures/iCAPs_k' + str(k1) + '_vs_atlas_mean_0_'

        print(f'METHOD : Comparing the datasets to the atlas for components at specific K values  at K = {k1}')
        data=nib.load(self.img_f).get_fdata()
        map_order = self._sort_maps(data, sorting_method=sorting_method) # The 1st dataset is sorted
        data_sorted = data[:,:,:,map_order]

        atlas_data=nib.load(self.atlas_f).get_fdata()
        atlas_map_order = self._sort_maps(atlas_data, sorting_method=sorting_method) # The 1st dataset is sorted
        atlas_data_sorted = atlas_data[:,:,:,atlas_map_order]


        # Compute the similarity coefficient and its mean for either selected method : 'Cosine', 'Dice', 'Euclidean distance', 'Overlap'
        if similarity_method == 'Cosine':
            if mask == None :
                mask=nib.load(self.config["project_dir"] + self.config["templates"]["spinalcord"]["cord"]).get_fdata()
                        
            else:
                mask=nib.load(mask).get_fdata()

            similarity_matrix,_, orderY = self._compute_similarity(data_sorted, atlas_data_sorted, mask1=mask, mask2=mask, thresh1=thr, thresh2=1, method=similarity_method, match_compo=True, verbose=False)
              
        else:
            similarity_matrix,_, orderY = self._compute_similarity(data_sorted,  atlas_data_sorted,  thresh1=thr, thresh2=1, method=similarity_method, match_compo=True, verbose=False)
        
        mean_similarity = mean(x for x in np.diagonal(similarity_matrix) if x !=-1) # If ks are different, we avoid taking -1 values (no correspondance)
 


        # Plot similarity matrix
        print(np.diagonal(similarity_matrix))
        plt.figure(figsize=(7,7))
        sns.heatmap(similarity_matrix, linewidths=.5, square=True, cmap='YlOrBr', vmin=0, vmax=1, xticklabels=orderY+1, yticklabels=np.array(range(1,k1+1)),cbar_kws={'shrink' : 0.8, 'label': similarity_method});
        plt.xlabel("Atlas")
        plt.ylabel("iCAPs")
            
        print(f'The mean similarity is {mean_similarity:.2f}' + " ± " + str(np.round(stdev((x for x in np.diagonal(similarity_matrix) if x !=-1)),2)))
            

            
        # Save results
        if save_results == True:
            if self.load_subjects != True:
                np.savetxt(output_fname +'.txt',mean_similarity)
                   
        # Save figure  
        if save_figure == True:
            plt.savefig(output_fig + str(round(mean_similarity*100)) + ".pdf", format='pdf')  

    def _compute_similarity(self, data1, data2, mask1=None, mask2=None, thresh1=1.6,thresh2=1, method='Cosine', match_compo=False, verbose=False):
        # Number of components is equal to the max between the two sets
        k = np.max([data1.shape[3],data2.shape[3]]) # Save number of components for later use, shape 3 = number of components
        if method == 'Dice' or method == 'Overlap' or method == 'Euclidean distance' or method == 'Euclidean distance abs' : # Binarize data if needed
            data1_bin = np.where(data1 >= thresh1, 1, 0)
            data2_bin = np.where(data2 >= thresh2, 1, 0)

        elif method == 'Cosine': # Prepare structures to save vectorized maps if needed
            if mask1 is None or mask2 is None: # Check if masks have been given
                raise(Exception(f'The "Cosine" method requires masks as inputs!'))
            mask1_vec = np.reshape(mask1,(mask1.shape[0]*mask1.shape[1]*mask1.shape[2],1)) # Reshape masks
            mask2_vec = np.reshape(mask2,(mask2.shape[0]*mask2.shape[1]*mask2.shape[2],1)) 
            if np.count_nonzero(mask1_vec) != np.count_nonzero(mask2_vec):
                # If data shapes are different, we take the largest mask.
                if np.count_nonzero(mask1_vec) > np.count_nonzero(mask2_vec): 
                    mask2_vec = mask1_vec
                else:
                    mask1_vec = mask2_vec

            data1_vec = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],k))
            data2_vec = np.zeros((data2.shape[0]*data2.shape[1]*data2.shape[2],k))
            data1_masked = np.zeros((np.count_nonzero(mask1_vec),k))
            data2_masked = np.zeros((np.count_nonzero(mask2_vec),k))    
            
        if verbose == True:
            print(f"...Compute similarity between pairs of components")
        similarity_matrix = np.zeros((k,k))
            
        for k1 in range(0,k):
            if method == 'Cosine': # Reshape as vector & mask if needed 
                if k1 < data1.shape[3]: # Check that k is included in the data
                    data1_vec[:,k1] = np.reshape(data1[:,:,:,k1],(data1.shape[0]*data1.shape[1]*data1.shape[2],))
                    data1_masked[:,k1] = data1_vec[np.flatnonzero(mask1_vec),k1]
            
            for k2 in range(0,k):
                if method == 'Dice' or method == 'Overlap':
                    # For the intersection, we multiply the two binary maps and count the number of elements
                    if k1 < data1.shape[3] and k2 < data2.shape[3]: # If the element exist in both datasets, we compute the similarity
                        nb_el_inters = np.sum(np.multiply(data1_bin[:,:,:,k1], data2_bin[:,:,:,k2])) 
                        nb_el_1 = np.sum(data1_bin[:,:,:,k1])
                        nb_el_2 = np.sum(data2_bin[:,:,:,k2])
                        if method == 'Dice':
                            similarity_matrix[k1,k2] = 2*nb_el_inters / (nb_el_1+nb_el_2)
                        elif method == 'Overlap':
                            if nb_el_1 > nb_el_2:
                                similarity_matrix[k1,k2] = nb_el_inters / (nb_el_2)
                            elif nb_el_2 > nb_el_1:
                                similarity_matrix[k1,k2] = nb_el_inters / (nb_el_1)
                            elif nb_el_2 == nb_el_1:
                                similarity_matrix[k1,k2] = 2*nb_el_inters / (nb_el_1+nb_el_2)
                                print("the two cluster are equal, 'Dice' methods was applied instead of 'Dice_smaller'")
                            similarity_matrix[k1,k2] = similarity_matrix[k1,k2]*100
                        
                    else: # Else, we just set it to -1
                        similarity_matrix[k1,k2] = -1
                elif method == 'Euclidean distance' or method == 'Euclidean distance abs' :
                    if k1 < data1.shape[3] and k2 < data2.shape[3]: # If the element exist in both datasets, we compute the similarity
                        # Label data to find the different clusters
                        lbl1 = label(data1_bin[:,:,:,k1])[0]
                        lbl2 = label(data2_bin[:,:,:,k2])[0]
        
                        # We calculate the center of mass of the largest clusters
                        cm1 = center_of_mass(data1_bin[:,:,:,k1],lbl1,Counter(lbl1.ravel()).most_common()[1][0])
                        cm2 = center_of_mass(data2_bin[:,:,:,k2],lbl2,Counter(lbl2.ravel()).most_common()[1][0])
                        
                        if method == 'Euclidean distance abs':
                            # inverse of the euclidean distance between CoG
                            #similarity_matrix[k1,k2]=1/(np.mean(np.abs(np.array(cm1)-np.array(cm2)))) 
                            
                        # similarity_matrix[k1,k2] = np.abs([float(cm1[2])-float(cm2[2])])*0.5
                            similarity_matrix[k1,k2] = 1/np.abs([float(cm1[2])-float(cm2[2])])*0.5

                        elif method == 'Euclidean distance':
                            similarity_matrix[k1,k2] = np.mean([float(cm1[1])-float(cm2[1]),float(cm1[2])-float(cm2[2])])
                    else:
                        similarity_matrix[k1,k2] = -1
                elif method == 'Cosine':
                    data2_vec[:,k2] = np.reshape(data2[:,:,:,k2],(data2.shape[0]*data2.shape[1]*data2.shape[2],)) # Vectorize
                    data2_masked[:,k2] = data2_vec[np.flatnonzero(mask2_vec),k2]
                    if k1 < data1.shape[3] and k2 < data2.shape[3]: # If the element exist in both datasets, we compute the similarity
                        similarity_matrix[k1,k2] = cosine_similarity(data1_masked[:,k1].reshape(1, -1), data2_masked[:,k2].reshape(1, -1))
                    else:
                        similarity_matrix[k1,k2] = -1
                else:
                    raise(Exception(f'The method {method} has not been implemented'))
            
        if match_compo == True:
            if verbose == True:
                print(f"...Ordering components based on maximum weight matching")
            orderX,orderY=scipy.optimize.linear_sum_assignment(similarity_matrix,maximize=True)
            # if the same composantes match
            similarity_matrix = similarity_matrix[:,orderY]
        else:
            orderX = np.array(range(0,k))
            orderY = np.array(range(0,k))

               
        if verbose == True:
            print(f"DONE!")

        return similarity_matrix, orderX, orderY
    
    def _sort_maps(self,data, sorting_method,threshold=None):
        ''' Sort maps based on sorting_method (e.g., rostrocaudally)
        
        Inputs
        ----------
        data : array
            4D array containing the k maps to order  
        sorting_method : str
            Method used to sort maps (e.g., 'rostrocaudal', 'rostrocaudal_CoM', 'no_sorting')
        threshold: str
            put a threshold if you're using the 'rostrocaudal_CoM' method
        Output
        ----------
        sort_index : list
            Contains the indices of the sorted maps   
        '''  
        if sorting_method == 'rostrocaudal':
            print('Sorting method: rostrocaudal (max value)')
            max_z = []; 
            for i in range(0,data.shape[3]):
                max_z.append(int(np.where(data == np.nanmax(data[:,:,:,i]))[2][0]))  # take the first max in z direction      
            sort_index = np.argsort(max_z)
            sort_index= sort_index[::-1] # Invert direction to go from up to low
        elif sorting_method == 'rostrocaudal_CoM':
            print('Sorting method: rostrocaudal (center-of-mass biggest cluster)')
            cm_z=[]
            data_thresh =  np.where(data > threshold, data, 0) # Threshold data
            
                
            # We calculate the center of mass of the largest clusters
            for i in range(0,data.shape[3]):
                # Label data to find the different clusters
                lbl1 = label(data_thresh[:,:,:,i])[0]
                cm = center_of_mass(data_thresh[:,:,:,i],lbl1,Counter(lbl1.ravel()).most_common()[1][0]) # Take the center of mass of the larger cluster
                cm_z.append(cm[2])
            
            sort_index = np.argsort(cm_z)
            sort_index= sort_index[::-1] # Invert direction to go from up to low
                
        elif sorting_method == 'no_sorting':
            print('Sorting method: no_sorting')
            sort_index = list(range(data.shape[3]))
        else:
            raise(Exception(f'{sorting_method} is not a supported sorting method.'))
        return sort_index
