# -*- coding: utf-8 -*-
import os, glob, math
import numpy as np
import nibabel as nib
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt


#nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import non_parametric_inference
from nilearn.maskers import NiftiMasker
from nilearn.glm import threshold_stats_img
from nilearn import plotting, image

from scipy import stats
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="Random effects covariance is singular",
    category=UserWarning,
    module="statsmodels.regression.mixed_linear_model"
)
class Statistics:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, config,IDs=None,ana_dir=None,analysis="ts_features",structure="spinalcord"):
        self.config = config # load config info
        if IDs==None:
            self.IDs= self.config["participants_IDs_ALL"]
        else:
            self.IDs=IDs
        if ana_dir is None:
            self.outputdir= self.output_dir = self.config["project_dir"] +self.config[analysis]["analysis_dir"][structure]
        else:
            self.outputdir=ana_dir
        
        self.population_info=self.config["project_dir"] +config["population_infos"]
        self.metadata=pd.read_csv(self.population_info, delimiter='\t')
        self.metadata.set_index('participant_id', inplace=True)


    def compute_regional_coupling(self,IDs=None, matrix1=None, matrix2=None,metrics=["data1","data2"], df_out=False, metadata_df=None ):
        """
        Compute regional structure-function coupling (i.e, for each row) for each individual.
        
        Parameters
        ----------
        IDs : list of str
            Sorted list of subject IDs.
        matrix1 : list of np.ndarray
            Matrices (shape n_individuals x n_nodes x n_nodes).
        matrix2 : list of np.ndarray
            Matrices (shape n_individuals x n_nodes x n_nodes).
        df_out: bool
            Wether to export a dataframe (True or False)
        metadata_df: dataframe
            metadata should be provide if df_out=True


        Returns
        -------
        List of Spearman correlation coefficients per subject.
        Or
        Dataframe with spearm correlation coefficients and metadata information
        """
        if IDs==None:
            IDs=self.IDs
        
        print()
        if matrix1 is None or matrix2 is None:
            raise ValueError("Please provide a list of np.narray for each matrix (shape n_indiviudals x n_nodes x n_nodes)")

        n_nodes = matrix1[0].shape[0]
        regional_coupling = np.full((len(IDs), n_nodes), np.nan)
        all_data = {metrics[0]: [], metrics[1]: []}

        ID_coupling = []
        for ID_nb, ID in enumerate(IDs):
            m1=matrix1[ID_nb]
            m2=matrix2[ID_nb]
            for node_i in range(n_nodes):
                mask=np.ones(n_nodes,dtype=bool)
                mask[node_i]= False # exclude self

                m1_vec=m1[node_i,mask]
                m2_vec=m2[node_i,mask]

                valide=np.isfinite(m1_vec) & np.isfinite(m2_vec) 


                r, _ =spearmanr(m1_vec[valide],m2_vec[valide])
                regional_coupling[ID_nb,node_i]=r
                ID_coupling.append(ID)

        #mean over the nodes
        m1_all = np.stack(matrix1); m1_mean=np.nanmean(m1_all,axis=0)
        m2_all = np.stack(matrix2); m2_mean=np.nanmean(m2_all,axis=0)

        all_nodes_m1=[];all_nodes_m2=[]
        for i in range(n_nodes):
            m1_vec = m1_mean[i, :]
            m2_vec = m2_mean[i, :]
            mask = np.ones(n_nodes, dtype=bool)
            mask[i] = False
            m1_vec = m1_vec[mask]
            m2_vec = m2_vec[mask]
            
            # Filter valid entries
            valid = np.isfinite(m1_vec) & np.isfinite(m2_vec)
            all_nodes_m1.append(m1_vec[valid])
            all_nodes_m2.append(m2_vec[valid])

        all_data[metrics[0]]=np.concatenate(all_nodes_m1)
        all_data[metrics[1]]=np.concatenate(all_nodes_m2)
        all_data = pd.DataFrame(all_data)
        if df_out:
            if metadata_df is None:
                raise ValueError("Please provide a metadata dataframe with 'participant_id', 'age' and 'sex' columns for each individual")
            else:
                data_list = []

            for ID_nb, ID in enumerate(IDs):
                age = metadata_df.loc[metadata_df["participant_id"] == ID, "age"].values[0]
                sex = metadata_df.loc[metadata_df["participant_id"] == ID, "sex"].values[0]

                for node_i in range(n_nodes):
                    coupling = regional_coupling[ID_nb, node_i]
                    data_list.append({
                        "IDs": ID,
                        "age": age,
                        "sex": sex,
                        "node": node_i,
                        "regional_coupling": coupling
                    })

            df = pd.DataFrame(data_list)

        return (regional_coupling if df is None or df.empty else df), all_data


    
    def matrix_stats(self, array_list=None, output_dir=None, output_tag='', tail="two-sided", correction='fdr',
                 redo=False,stat_type="ttest", covariates=None, dependent_var=None):
        """
        Compute statistical tests (t-test or OLS) on individual correlation matrices.

        Parameters:
            array_lit: list containing individual-level matrices as np arrays.
            output_dir (str): Directory to save results.
            output_tag (str): Optional tag for output file naming.
            tail (str): Tail direction for t-test: "two-sided" or "greater".
            correction (str): Multiple comparison correction: "fdr" or "Bonferroni".
            redo (bool): Force re-run.
            stat_type (str): Type of statistical test: "ttest" or "ols".
            covariates (list): List of covariate names for OLS.
            dependent_var (str): Column name to be used as dependent variable (e.g., 'age').
        """

        # Load the matrix
        matrices = []
        for ID_nb, ID in enumerate(self.IDs):
            matrices.append(array_list[ID_nb])
        
        # load info about participant covariates
        if covariates is not None:
            covariate_data = {}
            
            for covariate in covariates:
                covariate_data[covariate] = self.metadata.loc[self.IDs, covariate]
        # Info about the matrix
        seeds = range(matrices[0].shape[0])
        targets = range(matrices[0].shape[1])
        overall_stat = np.full((len(seeds), len(targets)), np.nan)
        overall_p = np.full((len(seeds), len(targets)), np.nan)


        for i, seed in enumerate(seeds):
            for j, target in enumerate(targets):
                values = [mat[seed, target] for mat in matrices if mat[seed, target] != 0]
                subj_ids = [ID for idx, ID in enumerate(self.IDs) if matrices[idx][seed, target] != 0]
                n = len(values)
                if n < 2:
                    continue

                if stat_type == "ttest":
                    res = stats.ttest_1samp(values, popmean=0, alternative=tail)
                    overall_stat[i, j] = res.statistic
                    overall_p[i, j] = res.pvalue

                elif stat_type == "ols" or dependent_var is None:
                    if covariates is None :
                        raise ValueError("For OLS, both covariates and dependent_var must be provided")

                    values_series = pd.Series(values, index=subj_ids, name='value')
                    sub_df = self.metadata.loc[subj_ids, covariates]
                    data = pd.concat([values_series, sub_df], axis=1).dropna()

                    if data.shape[0] < 3:
                        continue

                    y = data['value'].astype(float).values

                    if len(covariates) > 1:
                        X_full = pd.get_dummies(data[covariates], drop_first=True).astype(float)
                        X_full = sm.add_constant(X_full)
                        X_reduced = sm.add_constant(np.ones((len(data), 1)))

                        model_full = sm.OLS(y, X_full).fit()
                        model_reduced = sm.OLS(y, X_reduced).fit()

                        # Compute R squared values
                        r2_full = model_full.rsquared
                        r2_reduced = model_reduced.rsquared
                        partial_r2 = r2_full - r2_reduced # Partial R squared
                        

                        # Find sign of the dependent_var coefficient(s)
                        coef_names = model_full.params.index.tolist()
                        coef_signs = []
                        for cname in coef_names:
                            if cname == dependent_var or cname.startswith(dependent_var + "_"):
                                coef_signs.append(np.sign(model_full.params[cname]))
                        sign = coef_signs[0] if coef_signs else 0
                        signed_partial_r2 = partial_r2 * sign


                        overall_stat[i, j] = signed_partial_r2
                        overall_p[i, j] = model_full.pvalues.get(dependent_var, np.nan)


                    else:
                        X = pd.get_dummies(data[covariates], drop_first=True).astype(float)
                        X = sm.add_constant(X)
                        model = sm.OLS(y, X).fit()
                        covariate_name = model.params.index[1]

                        overall_stat[i, j] = model.params[covariate_name]
                        overall_p[i, j] = model.pvalues[covariate_name]

                        model = sm.OLS(y, X).fit()

        if correction:
            if correction == "fdr":
                _, pvals_corr, _, _ = multipletests(pvals_flat[mask], method='fdr_bh')
            elif correction == "Bonferroni":
                _, pvals_corr, _, _ = multipletests(pvals_flat[mask], method='bonferroni')
            else:
                pvals_corr = pvals_flat[mask]
            
            corrected = np.full_like(pvals_flat, np.nan)
            corrected[mask] = pvals_corr
            corrected_matrix = corrected.reshape(overall_p.shape)
            pval_corrected_df = pd.DataFrame(corrected_matrix, index=seeds, columns=targets)



        return overall_stat,overall_p
    
    def signed_partial_r2(self,df=None,y=None, predictor=None,covariates=None,random=None ):

        """

        Compute signed partial R² of age controlling for sex.
        df: df containing y, covariates and dependent_var information
        y:       string, name of the columns for y (ex: "corr")
        predictor: string, name of the columns for dependent_var (main X, ex:e "age")
        covariates: list of string, name of the columns for covariates (additional covariate ex: ["sex"])
        
        
        Returns:
        signed_r2: float, signed partial R² (NaN if non-significant or no variance)
        p_val:     p-value for age term
        beta:      age coefficient
        """
        # Check covariates are numerics:
  
        covariates_bin=[]

        for cov in covariates:
            s_num = pd.to_numeric(df[cov], errors='coerce')# Attempt to coerce to numeric
            
            # Case A: already binary numeric (e.g. 0/1 or 1/2)
            unique_vals = set(s_num.dropna().unique())
            if pd.api.types.is_numeric_dtype(s_num) and len(unique_vals) == 2:
                df[f'{cov}_bin'] = s_num.astype(int)
            
            # Case B: not binary numeric → categorical encoding
            else:
                cat = pd.Categorical(df[cov])
                codes = cat.codes
                
                df=df.copy()
                df.loc[:,f'{cov}_bin'] = codes
            covariates_bin.append(f'{cov}_bin')

        #print(df[['sex', 'sex_bin']].drop_duplicates())
        # full model
        if random is None:
            columns_needed = [y, predictor] + covariates_bin
            df_clean = df[columns_needed].dropna()
            
            X_full = sm.add_constant(df_clean[[predictor] + covariates_bin])
            m_full = sm.OLS(df_clean[y], X_full).fit()

            # reduced model
            X_red  = sm.add_constant(df_clean[covariates_bin])
            m_red  = sm.OLS(df_clean[y], X_red).fit()
        
        else:
            # Full model with random intercept for subject ID
            formula_full = f"{y} ~ {predictor} + {' + '.join(covariates_bin)}"
            m_full = smf.mixedlm(formula_full, df, groups=df[random]).fit()

            # Reduced model with random intercept for subject ID
            formula_reduced = f"{y} ~ {' + '.join(covariates_bin)}"
            m_red = smf.mixedlm(formula_reduced, df, groups=df[random]).fit()

        # test significance of age
        p_age = m_full.pvalues[predictor]
        beta_age  = m_full.params[predictor]
        stat_age  = m_full.tvalues[predictor]
        beta_sex  = m_full.params['sex_bin']
        stat_sex  = m_full.tvalues['sex_bin']
        p_sex  = m_full.pvalues['sex_bin']
        
        # compute SS
        if random is None:
            ss_full    = np.sum(m_full.resid ** 2)
            ss_reduced = np.sum(m_red.resid ** 2)
            ss_total   = np.sum((df[y] - df[y].mean())**2)

            # partial R²
            partial_r2 = (ss_reduced - ss_full) / ss_total

            # signed
            signed_r2 = np.sign(beta_age) * partial_r2
        else:
            signed_r2 = "_"#m_full.rsquared - m_red.rsquared
        
        return signed_r2, p_age, p_sex, beta_age, beta_sex, stat_age, stat_sex


class fMRI_Stats:
    def __init__(self, config,ana_name,output_tag='GLM',seed_structure='spinalcord',target_structure='brain',outputdir=None,save_ana=False):
        '''
        The Stats class will initiate the group level analysis
        I. Initiate variable
        II. Create output directory (if save_ana=True)
        III. Select first level data
        
        Attributes
            ----------
        config : dict
        measure: str
        first level measure could be "MI" or "Corr"
        '''
        
        #>>> I. Initiate variable -------------------------------------
        self.config = config # load config info
        self.ana_name=ana_name
        self.measure=self.config["statistics"]["measure"]
        self.model=self.config["statistics"]["model_stats"] ##OneSampleT #TwoSampT_paired or #TwoSampT_unpaired or "HigherOrder_paired"
        self.mask_img=self.config["main_dir"]+self.config["statistics"]["mask"][target_structure]# if no mask was provided the whole target image will be used
        self.seed_structure=seed_structure
        self.seed_names=self.config["seeds"][self.seed_structure]["names"] # seeds to include in the analysis
        self.IDs= self.config["participants_IDs"]

        
        if "targets" in self.config:
            self.target=self.config["targets"][self.target_structure]["names"][0]
            self.target_structure=target_structure
        else:
            self.target=self.config["seeds"][self.seed_structure]["names"][0]
            self.target_structure=seed_structure

        if outputdir:
            self.main_outputdir=outputdir
        else:
            if target_structure in self.config["analysis_dir"]:
                self.main_outputdir= self.config["main_dir"] +self.config["analysis_dir"][target_structure]
            else:
                self.main_outputdir= self.config["main_dir"] +self.config["analysis_dir"]
        
        

        self.output_tag=self.ana_name
        print("************************************** ")
        print("Initiate " + self.ana_name + " analysis")
        print("  ")
        print("> Statistical model: " +self.model)
        print("> Number of participants: "+ str(len(self.IDs)))
        print("> Mask : " + os.path.basename(self.mask_img))
        
        # check if the right number of seed is provided for the statistical analysis
        for seed_name in self.seed_names:
            if self.model=="OneSampleT" and len(self.seed_names)!=1:
                raise ValueError(">>>> Only One seed should be provided for One sample t-test or try 'TwoSampT_paired' or 'TwoSampT_unpaired' or 'HigherOrder_paired'")
            elif self.model=="TwoSampT_paired" and len(self.seed_names)!=2 or self.model=="TwoSampT_unpaired" and len(self.seed_names)!=2:
                raise ValueError(">>>> Two seeds should be provided for Two sample t-test or try 'OneSampleT' or 'HigherOrder_paired'")


                
        #>>> II. Create output directory (if save_ana=True) -------------------------------------
        if save_ana==True:
            self.output_dir=self.main_outputdir + '/2_second_level/'+output_tag+'/'+self.model+'/'+ os.path.basename(self.mask_img).split(".")[0] +'/' +self.measure +"/" + self.output_tag # name of the output dir
     
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir,exist_ok=True)  # Create output directory:
            print("> Saved here : " + self.output_dir)
            print("  ")
        

        #>>> III. Select first level data: -------------------------------------
        self.data_1rstlevel={};self.data_1rstlevel_4D={};
        for seed_name in self.seed_names:
            self.data_1rstlevel[seed_name]=[];self.data_1rstlevel_4D[seed_name]=[];
            for ID_nb,ID in enumerate(self.IDs):
                ID_name='sub-' +  ID
                tag_files = {"Corr": "corr","MI": "mi"}

                if self.measure=="MI" or self.measure=="Corr":
                    self.tag_file = tag_files.get(self.measure, None)
                    self.data_1rstlevel[seed_name].append(glob.glob(self.main_outputdir+self.config["first_level"] +'/'+ seed_name+'/'+ self.target +"_fc_maps/" +self.tag_file + '/'+ self.tag_file+ "_"+ ID_name + "*.nii.gz")[0])
                
                elif self.measure=="alff":
                    self.tag_file='s_Z'
                    #print(self.main_outputdir+self.config["first_level"] +'/'+ ID_name + "*" + self.tag_file +'.nii.gz')
                    self.data_1rstlevel[seed_name].append(glob.glob(self.main_outputdir+self.config["first_level"] +'/'+ ID_name + "*" + self.tag_file +'.nii.gz')[0])
                
                elif self.measure=="MTR":
                    self.tag_file='masked_MTR'
                    preprocess_dir=self.config["preprocess_dir"]["main_dir"] if ID[0]=="A" else self.config["preprocess_dir"]["bmpd_dir"]
                    self.data_1rstlevel[seed_name].append(glob.glob(preprocess_dir+self.config["PAM50_space"]["MTR"].format(ID))[0])
                


            if self.measure=="MI" or self.measure=="Corr":
                self.data_1rstlevel_4D[seed_name]=glob.glob(self.main_outputdir+self.config["first_level"] +'/'+ seed_name+'/'+ self.target +"_fc_maps/" + self.tag_file+ '/'+ self.tag_file + "_n"+ str(len(self.IDs)) +"*" +seed_name + "_4d.nii*")[0]
            elif self.measure=="alff":
                    self.data_1rstlevel_4D[seed_name]=glob.glob(self.main_outputdir+self.config["second_level"] +'/*alff_*' + "_n"+ str(len(self.IDs))+ '_'+ self.tag_file+   "_4d.nii.gz")[0]

            print(seed_name)
         

    def design_matrix(self,contrast_name=None,covariate=None,plot_matrix= False,save_matrix=False):
        '''
        Create and plot the design matrix for "OneSampleT" or  "TwoSampT_unpaired" or "TwoSampT_paired"
        For one matrix per contrast is created
        
        Attributes
        ----------
        contrast_name: str 
        
        plot_matrix: bool (default: False)
            To plot the design matrix
        
        save_matrix: default: False
            To save the design matrix. If True then plot_matrix will be turn True
            output: 
                - design_matrix_*.png : Save the image of the desgin matrix
                design_matrix_*.npy : Save the desgin matrix in numpy format
                
        
        To add: Contrats: dict (default : None)
            it will not be used for "OneSampleT" or  "TwoSampT_unpaired" or "TwoSampT_paired" models
            
            A dictonnary should be provided for complexe models "HigherOrder_paired"
            Contrast={"nameofthecontrast1": contrast1_values,
            "nameofthecontrast2": contrast2_values}
            contrast_values shape: (number of functional images), exemple: [1,1,0,0] for 4 functional images
        
        Return
        ----------
        Design_matrix:  dict
        Contained one matrix for each contrast (1: matrix for OnSample T and 4 matrices for TwoSampT )
        '''
        
        #>>>>>>>>  Initiate variables
        self.contrast_name=contrast_name
        if save_matrix==True:
            plot_matrix==True # matrix cannot be saved if them has not been plot
        
        IDs_nb=len(self.IDs) # indicate the totalnumber of participant to analyse
        contrasts={}; # create empty dict
        Design_matrix={} # create empty dict
                    
        #>>>>>>>> Create contrasts for each kind of test ____________________
        contrasts=self._generate_contrast()
        #>>>>>>>> Create a desgin matrix for each kind of test ____________________
        # For un unpaired tests:
        if self.model=="OneSampleT" or self.model=="TwoSampT_unpaired" :
            for i, (contrast,values) in enumerate(contrasts.items()):
                if covariate:
                    covariate = np.array(covariate) if isinstance(covariate, list) else covariate
                    intercept = np.ones(IDs_nb)
                    if covariate.shape[0] == 2:
                        Design_matrix[contrast]=pd.DataFrame(np.vstack((covariate[0],covariate[1],intercept)).T, columns=["cov1","cov2", "intercept"],)
                    else:
                        Design_matrix[contrast]=pd.DataFrame(np.vstack((covariate,intercept)).T, columns=["cov1", "intercept"],)
                else:
                    Design_matrix[contrast]=pd.DataFrame(np.hstack((values[:, np.newaxis])), columns=[contrast])
        
        # For paired tests:                            
        elif self.model=="TwoSampT_paired" or self.model=="HigherOrder_paired":
            contrasts=self._generate_contrast()
            
            # Add participant effect:
            for i, (contrast,values) in enumerate(contrasts.items()):
                Design_matrix[contrast]=pd.DataFrame(np.hstack((values[:, np.newaxis],np.concatenate([np.eye(IDs_nb)] * len(self.seed_names), axis=0))), columns=[contrast] + self.IDs)
            

        #Plot the matrix
        ### Create a subplot per contrast
       
        if plot_matrix== True:
            if self.model=="OneSampleT" or self.model=="TwoSampT_unpaired" :
                num_subplots = len(Design_matrix)# Define the number of subplots and their layout
                fig, axes = plt.subplots(1, num_subplots ,figsize=(len(Design_matrix)*2,5)) # # Create a figure and subplots
            
            elif self.model=="TwoSampT_paired" or self.model=="HigherOrder_paired":
                if len(self.seed_names) <5:
                    num_subplots = int(len(Design_matrix)) # Define the number of subplots and their layout
                    fig, axes = plt.subplots(1, num_subplots ,figsize=(len(Design_matrix)*4,5)) # # Create a figure and subplots
                else:  
                    num_subplots = int(math.ceil(len(Design_matrix)/2)) # Define the number of subplots and their layout
                    fig, axes = plt.subplots(2, num_subplots ,figsize=(len(Design_matrix)*3,10)) # # Create a figure and subplots

            if num_subplots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
                
            # Loop over the subplots and plot the data
            for i, (title,values) in enumerate(Design_matrix.items()):
                plot_design_matrix(values,ax=axes[i])
                axes[i].set_ylabel("maps")
            fig.suptitle("Desgin matrix " + self.model,fontsize=12)
            plt.tight_layout() # Adjust the layout and spacing of subplots
            
            
            if save_matrix==True:
                if not os.path.exists(self.output_dir):
                     raise ValueError(">>>> " +self.output_dir+ " directory should be created first with Stats() function")
                
                plt.savefig(self.output_dir + "/design_matrix_" + self.output_tag + ".png")
                np.save(self.output_dir + "/design_matrix_" + self.output_tag + ".npy",Design_matrix)
                
            plt.show() # Display the figure
                
        return Design_matrix
    
    
    
    def _generate_contrast(self):
        
        contrast_name=self.contrast_name
        contrasts={}
        
        if self.model== "OneSampleT" and contrast_name==None: 
            contrasts["Main " + self.seed_names[0]]=np.hstack(([1] * len(self.IDs)))
            
        elif self.model=="TwoSampT_paired" or self.model=="TwoSampT_unpaired" :
            contrasts["Main_" + self.seed_names[0]]=np.hstack(([1] * len(self.IDs), [0] * len(self.IDs))) # Main contrast for the first factor
            contrasts["Main_" + self.seed_names[1]]=np.hstack(([0] * len(self.IDs), [1] * len(self.IDs))) # Main contrast for the second factor
            contrasts[self.seed_names[0] + "_vs_" + self.seed_names[1]]=np.hstack(([1] * len(self.IDs), [-1] * len(self.IDs))) # contrast between the two
            contrasts[self.seed_names[1] + "_vs_" + self.seed_names[0]]=np.hstack(([-1] * len(self.IDs), [1] * len(self.IDs))) # contrast between the two
            contrasts["All effect"]=np.hstack(([1] * len(self.IDs), [1] * len(self.IDs)))
        
        
        elif len(self.seed_names)==4 and contrast_name==None:
            #VR, VL, DR, DL
            for seed_nb in range(0,len(self.seed_names)):
                contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.IDs) * seed_nb, [1] * len(self.IDs), [0] * len(self.IDs)* (len(self.seed_names)-(seed_nb+1))))
            contrasts["Ventral Effect"]=np.hstack(([1] * len(self.IDs) *2, [0] * len(self.IDs) * 2))
            contrasts["Dorsal Effect"]=np.hstack(([0] * len(self.IDs)*2, [1] * len(self.IDs) * 2))
            contrasts["Right Effect"]=np.hstack((np.tile(([[1] * len(self.IDs)+ [0] * len(self.IDs)]),2)))
            contrasts["Left Effect"]=np.hstack((np.tile(([[0] * len(self.IDs)+ [1] * len(self.IDs)]),2)))
            contrasts["Ventral vs Dorsal"]=np.hstack(([1] * len(self.IDs) *2, [-1] * len(self.IDs) * 2))
            contrasts["Dorsal vs Ventral"]=np.hstack(([-1] * len(self.IDs)*2, [1] * len(self.IDs) * 2))
            contrasts["Right vs Left"]=np.hstack((np.tile(([[1] * len(self.IDs)+ [-1] * len(self.IDs)]),2)))
            contrasts["Left vs Righ"]=np.hstack((np.tile(([[-1] * len(self.IDs)+ [1] * len(self.IDs)]),2)))
            contrasts["All effect"]=np.hstack(([1] * len(self.IDs) * len(self.seed_names)))
                    
        elif len(self.seed_names)>4 and contrast_name==None:
            for seed_nb in range(0,len(self.seed_names)):
                contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.IDs) * seed_nb, [1] * len(self.IDs), [0] * len(self.IDs)* (len(self.seed_names)-(seed_nb+1))))
            contrasts["All effect"]=np.hstack(([1] * len(self.IDs) * len(self.seed_names)))
            
        elif contrast_name=="4quad_9levels":
            #for seed_nb in range(0,len(self.seed_names)):
                #contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.IDs) * seed_nb, [1] * len(self.IDs), [0] * len(self.IDs)* (len(self.seed_names)-(seed_nb+1))))
            contrasts["Ventral Effect"]=np.hstack(([1] * len(self.IDs) *9*2, [0] * len(self.IDs) * 9*2))
            contrasts["Dorsal Effect"]=np.hstack(([0] * len(self.IDs)*9*2, [1] * len(self.IDs) * 9*2))
            contrasts["Right Effect"]=np.hstack((np.tile(([[1] * len(self.IDs)+ [0] * len(self.IDs)]),9*2)))
            contrasts["Left Effect"]=np.hstack((np.tile(([[0] * len(self.IDs)+ [1] * len(self.IDs)]),9*2)))
            contrasts["Ventral vs Dorsal"]=np.hstack(([1] * len(self.IDs) *9*2, [-1] * len(self.IDs) * 9*2))
            contrasts["Dorsal vs Ventral"]=np.hstack(([-1] * len(self.IDs)*9*2, [1] * len(self.IDs) * 9*2))
            contrasts["Right vs Left"]=np.hstack((np.tile(([[1] * len(self.IDs)+ [-1] * len(self.IDs)]),9*2)))
            contrasts["Left vs Right"]=np.hstack((np.tile(([[-1] * len(self.IDs)+ [1] * len(self.IDs)]),9*2)))
            contrasts["All effect"]=np.hstack(([1] * len(self.IDs) * len(self.seed_names)))
            
        elif contrast_name=="D-R_9levels":
            for seed_nb in range(0,len(self.seed_names)):
                contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.IDs) * seed_nb, [1] * len(self.IDs), [0] * len(self.IDs)* (len(self.seed_names)-(seed_nb+1))))
            contrasts["Right Effect"]=np.hstack((np.tile(([[1] * len(self.IDs)+ [0] * len(self.IDs)]),9)))
            contrasts["Left Effect"]=np.hstack((np.tile(([[0] * len(self.IDs)+ [1] * len(self.IDs)]),9)))
            contrasts["Right vs Left"]=np.hstack((np.tile(([[1] * len(self.IDs)+ [-1] * len(self.IDs)]),9)))
            contrasts["Left vs Right"]=np.hstack((np.tile(([[-1] * len(self.IDs)+ [1] * len(self.IDs)]),9)))
            contrasts["All effect"]=np.hstack(([1] * len(self.IDs) * len(self.seed_names)))
                    
        
        
        return contrasts


    def secondlevelmodel(self,Design_matrix,parametric=True,estimate_threshold=False,plot_2ndlevel=False,save_img=False):
        '''
        This function calculate the second level model
        
        To compute a non-parametric analysis with automatic threshold estimation:
        1- Run a parametric analysis to obtnain z-map with low computational time
        2- Run non parametric analysis with estimate_threshold=True
        
        Attributes
        ----------
        Design_matrix: Dict
            The information about the contrasts. Contained one matrix for each contrast
            
        plot_2ndlevel: bool (default: False)
            To plot the uncorrected zmaps for each contrast
        
        save_img: bool, default: False
            To save the uncorrected maps for each contrast and each stats (.nii.gz)
        
        Estimate_threshold:  bool, default: False
            This option is useful if nonparametric measure is implemented (parametric=False).
        
 
        
        https://nilearn.github.io/dev/modules/generated/nilearn.glm.compute_contrast.html
        
        Return
        ----------
        contrast_map:  nifti images
        - 'z_score': z-maps
        - 'stat: t-maps
        - 'p_value': p-values maps
        - 'effect_size'
        - 'effect_variance'

        '''
        self.parametric=parametric
        # concatenates the files if there are multiple factors:
        input_files=[]
        for seed_name in self.seed_names:
            if len(self.seed_names)==1:
                input_files=self.data_1rstlevel[seed_name]
            elif len(self.seed_names)>1:
                input_files=np.concatenate((input_files,self.data_1rstlevel[seed_name]))
        
            
        # Load the nifti files
        nifti_files=[]
        print(input_files)
        for i in range(0,len(input_files)):
            nifti_files.append(nib.load(input_files[i]))

        # fit the model for each matrix and compute the constrast for parametrical statistics
       
        second_level_model={};contrast_map={}
        for i, (title,values) in enumerate(Design_matrix.items()):
            print(title)
   

            if parametric==True:
                second_level_model[title] = SecondLevelModel(mask_img=self.mask_img,smoothing_fwhm=None)
                second_level_model[title] = second_level_model[title].fit(nifti_files, design_matrix=Design_matrix[title])
                
                if 'cov1' in Design_matrix[title].columns:
                    contrast_map[title]=second_level_model[title].compute_contrast('cov1', second_level_stat_type="t",output_type="all")
                
                else:
                    contrast_map[title]=second_level_model[title].compute_contrast(title, second_level_stat_type="t",output_type="all")
      
                
                if save_img==True:
                    output_uncorr=self.output_dir + "/uncorr/"
                    if not os.path.exists(output_uncorr):
                        os.mkdir(output_uncorr)
                    nib.save(contrast_map[title]['z_score'], output_uncorr +"/zscore_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['stat'],output_uncorr + "/stat_" + title.split(" ")[-1] + ".nii.gz") # t or F statistical value
                    nib.save(contrast_map[title]['p_value'],output_uncorr + "/pvalue_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['effect_size'],output_uncorr + "/effectsize_" +title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['effect_variance'],output_uncorr+ "/effectvar_" + title.split(" ")[-1] + ".nii.gz")
                
                if plot_2ndlevel==True:
                        plotting.plot_glass_brain(
                            contrast_map[title]['stat'],
                            colorbar=True,
                            symmetric_cbar=False,
                            display_mode='lyrz',
                            threshold=z_thr,
                            vmax=5,
                            title=title)
            
            
            elif parametric==False:
                if estimate_threshold==True:
                    if not os.path.exists(self.output_dir + "/uncorr/"):
                        raise Exception("if estimate_threshold==True; parametric maps should be computed first to have z-sscored maps")
                    else:
                        output_uncorr=self.output_dir + "/nonparam/" # create output folder
                        if not os.path.exists(output_uncorr):
                            os.mkdir(output_uncorr)
                        z_thr,p_thr=self._estimate_threshold(self.output_dir +"/uncorr/zscore_" + title.split(" ")[-1] + ".nii.gz",5,self.output_dir + "/nonparam/")
                              
                else:
                    p_thr=0.05
                
                print(p_thr)
                contrast_map[title]=non_parametric_inference(nifti_files,
                                                             mask=self.mask_img,
                                                             design_matrix=Design_matrix[title],model_intercept=True,
                                                             n_perm=100, # should be set between 1000 and 10000
                                                             two_sided_test=False,
                                                             smoothing_fwhm=6,
                                                             tfce=True, # choose tfce=True or threshold is not None
                                                             threshold=p_thr,
                                                             n_jobs=8)
                if save_img==True:
                    output_corr=self.output_dir + "/nonparam/"
                    if not os.path.exists(output_corr):
                        os.mkdir(output_corr)
                    nib.save(contrast_map[title]['t'], output_corr +"/t_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['size'], output_corr +"/size_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['logp_max_t'],output_corr + "/logp_max_t_" + title.split(" ")[-1] + ".nii.gz") # t or F statistical value
                    nib.save(contrast_map[title]['logp_max_size'],output_corr + "/logp_max_size_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['mass'],output_corr + "/mass_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['logp_max_mass'],output_corr+ "/logp_max_mass_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['tfce'],output_corr + "/tfce_" + title.split(" ")[-1] + ".nii.gz")
                    nib.save(contrast_map[title]['logp_max_tfce'],output_corr+ "/logp_max_tfce_" + title.split(" ")[-1] + ".nii.gz")
                   
                    # Data are logp value logp=1.3 => p=0.05 ; logp=2 p=0.01 ; logp=3 p=0.001
                    string1="fslmaths " + output_corr + "/logp_max_size_" + title.split(" ")[-1] + ".nii.gz -thr 1.3 " +output_corr + "/logp_max_size_" + title.split(" ")[-1] + "_p-thr05.nii.gz"
                    string2="fslmaths " + output_corr + "/logp_max_size_" + title.split(" ")[-1] + ".nii.gz -thr 1.3 -bin " +output_corr + "/logp_max_size_" + title.split(" ")[-1] + "_p-thr05_bin.nii.gz"
                    
                    string3="fslmaths " + output_corr + "/t_" + title.split(" ")[-1] + ".nii.gz -mas " +output_corr + "/logp_max_size_" + title.split(" ")[-1] + "_p-thr05_bin.nii.gz " + output_corr + "/t_" + title.split(" ")[-1] + "_cluster-corr05.nii.gz"
                    
                    
                    print(string3)
                    os.system(string1);os.system(string2);os.system(string3);
                    
                if plot_2ndlevel==True:
                    if estimate_threshold==False:
                        z_thr=2.4

                    plotting.plot_glass_brain(
                            contrast_map[title]['size'],
                            colorbar=True,
                            symmetric_cbar=False,
                            display_mode='lyrz',
                            threshold=z_thr,
                            vmax=5,
                            title=title)

                       
        # fit the model for each matrix and compute the constrast for non-parametrical statistics
   
        return  contrast_map
    
    def secondlevel_grf_correction(self,z_map,ana_mask,z_thr=2.3,p_value=0.05,redo=False):

        #Create output dir
        output_grf=self.output_dir  + "/grf_corrected/"
        if not os.path.exists(output_grf):
            os.mkdir(output_grf)
        

        # compute smoothest to estimate dhl and volume
        smoothest_f=output_grf + '/smoothest_output.txt'
        if not os.path.exists(smoothest_f) or redo==True:
            str_sm=f"smoothest -z {z_map} -m {ana_mask} > {smoothest_f}"
            os.system(str_sm)

        # Read the output file to capture DLH and VOLUME
        with open(smoothest_f, "r") as file:
            output = file.read()

        # Extract DLH and VOLUME from the output
        dlh, volume = None, None
        for line in output.split("\n"):
            if "DLH" in line:
                dlh = line.split()[1]
            if "VOLUME" in line:
                volume = line.split()[1]

        print(f"DLH: {dlh}, VOLUME: {volume}")

        # run cluster calculation
        cluster_f=f"{output_grf}/cluster_z_thr{z_thr}"
        if not os.path.exists(cluster_f + '.nii.gz') or redo==True: 
            cluster_cmd = f"cluster -i {z_map} -t {z_thr} -p 0.05 -d {dlh} --volume={volume} -o {cluster_f}"
            os.system(cluster_cmd)

        # mask z_map by cluster corrected
        grf_corr_f=output_grf + '/' + os.path.basename(z_map).split('.')[0] + '_'+str(z_thr)+'_grf_corr.nii.gz'
        if not os.path.exists(grf_corr_f) or redo==True:
            
            mask_cmd=f"fslmaths {z_map} -mas {cluster_f}.nii.gz {grf_corr_f}"
            os.system(mask_cmd)

    def secondlevel_correction(self,maps,z_thr=4,p_value=0.05,cluster_threshold=10,corr=None,smoothing=None,plot_stats_corr=False,save_img=False,n_job=1):
        '''
        One sample t-test
        Attributes
        ----------
        maps: statisitcal maps from 2nd level fitting
        
        p_value : float or list, optional
        Number controlling the thresholding (either a p-value or q-value). Its actual meaning depends on the height_control parameter. This function translates p_value to a z-scale threshold. Default=0.001.

        corr: string, or None optional
        False positive control meaning of cluster forming threshold: None|’fpr’|’fdr’|’bonferroni’ Default=’fpr’.

        '''
        #if z_thr="auto":
            #definie trhreshold
        for i, (title,values) in enumerate(maps.items()):
            thresholded_map, threshold = threshold_stats_img(maps[title]["z_score"],
                                                             alpha=p_value,
                                                             threshold=z_thr,
                                                             height_control=corr,
                                                             cluster_threshold=cluster_threshold,
                                                             two_sided=False)
            


            if plot_stats_corr==True:
                thresholded_map=image.threshold_img(thresholded_map, 0,mask_img=self.mask_img, copy=True)

                self._plot_stats(thresholded_map, title ,threshold, cluster_threshold,vmax=4)


            if save_img==True:
                correction = corr if corr is not None else "nocorr"
                output_corrected=self.output_dir  + "/"+correction+"_corrected/"
                #print(output_corrected + "/" + title + "_" +corr+ "_p"+ str(p_value).split('.')[-1] +".nii.gz")
                if not os.path.exists(output_corrected):
                    os.mkdir(output_corrected)
                nib.save(thresholded_map, output_corrected + "/" + title + "_" +correction+ "_p"+ str(p_value).split('.')[-1] +".nii.gz")

    def _estimate_threshold(self,img,perc, output_dir):
        '''
        Attributes
        img: string
            input image
        perc: int
            type I error in percentage of voxels (default=5 => for a confidence level of 95%)
            
        Return
        Image of the voxel distribution
        z_thr: threshold in z-value
        p_value: threshold in p-value
        '''
        z_map=img#self.second_level_dir +"/uncorr/zscore_" + title + ".nii.gz"
        masker= NiftiMasker(self.mask_img, t_r=1.55,low_pass=None, high_pass=None) # seed masker
        vx_values=masker.fit_transform(z_map) #extract z values for each voxels
        percent=np.round(vx_values.shape[1]*perc/100) # the number of voxels to reach 5% of the total number of voxels
        vx_values_sorted=np.sort(vx_values) # sort voxel values for lower to higher values
        vx_values_perc=vx_values_sorted[0][int(vx_values.shape[1]-percent):vx_values.shape[1]] # select the last 5% (>95%)
        z_thr=vx_values_perc[0] # extract the z value >95%
        p_thr=2*(1 - norm.cdf(abs(z_thr), len(self.IDs)-1)) # convert into p value with a DOF= number of participants - 1

        # save the map of threshold selection
        mybins = np.linspace(np.min(vx_values), np.max(vx_values), 1000) # calculate the bin
        plt.axvline(x = vx_values_perc[0], color="red", label="stat threshold") # plot the threshold value in red
        plt.hist(vx_values_sorted[0][:],mybins) # plot the histogram of the distribution
        plt.legend() # plot the legend
        plt.savefig(output_dir + 'estimate_threshold.png')
        plt.title("z value distribution across voxels, thr= " + str(np.round(vx_values_perc[0],decimals=2)))
        plt.xlabel("z values")
        plt.ylabel("number of voxels")
        
        return z_thr, p_thr

                        
    def _plot_stats(self,second_level_map,title,threshold,cluster_threshold,vmax=5):
        '''
        Extract significant cluster in a table
        Attributes
        ----------
        stat_img : Niimg-like object
        Statistical image to threshold and summarize.

        stat_threshold float
        Cluster forming threshold. This value must be in the same scale as stat_img.

        '''
        display = plotting.plot_glass_brain(
            second_level_map,
            colorbar=True,
            vmax=vmax,
            display_mode='lyrz',
            title=title)  
        plotting.show()
