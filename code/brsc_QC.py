# Main imports ------------------------------------------------------------
import glob, os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from nilearn.image import math_img,smooth_img
from nilearn.input_data import NiftiMasker
from brsc_preprocess import Preprocess_Sc, Preprocess_Br
import seaborn as sns

def FD_fsl(config=None,ID=None,step=None,structure='brain',redo=False):
    '''
        This function calculate the FD using fsloutliers
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers

        Attributes:
        ----------
        config: load config file
        ID: participant ID
        step: specify the step if the preprocessing to select the image 'sct' or 'moco
        redo: put True to re-run the analysis on existing file (default=False)
    
    '''
    project="stratals" if ID[0]=="A" else "bmpd"
    if step==None:
        raise Warning("Step should be specify: 'stc' or 'moco' for now ")
    
    if step=="stc":
        folder=config["preproc_dir_"+project] + config["stc_files"]["dir"].format(ID,structure) 
        tag="sc" if structure=="spinalcord" else "brain"
        print(folder + config["stc_files"]["stc_f"].format(tag))
        i_img=glob.glob(folder + config["stc_files"]["stc_f"].format(tag))[0]
        mask=glob.glob(config["preproc_dir_"+project]+ config["seg_files"]["dir_mask"].format(ID,structure) +"/*mask.nii.gz")[0]
        

    elif step=="moco":
        folder=config["preproc_dir_"+project] + config["moco_files"]["dir"].format(ID,structure) 
        i_img=glob.glob(folder + config["moco_files"]["moco_f"])[0]
        mask=glob.glob(config["preproc_dir_"+project]+ config["seg_files"]["dir_mask"].format(ID,structure) +"/*mask.nii.gz")[0]
    
    else:
        raise Warning("Step should be specify: 'stc' or 'moco' for now ")

    
    o_file=folder +  "/fsloutliers/"+ "FD_fsloutliers_" + step+ "_"+structure 
    if not os.path.exists(folder +  "/fsloutliers/"):
        os.mkdir(folder +  "/fsloutliers/")
   
    if not os.path.exists(o_file +".txt") or redo==True:
        string="fsl_motion_outliers -i " + i_img +" -o " + o_file + ".txt -m " + mask+" -s " +o_file+".png --dvars --nomoco"
        print(string)
        os.system(string)


def tSNR(config=None,ID=None,i_img=None,mask=None,structure='brain',o_tag="_moco",inTemplate=False,redo=False):
    '''
        This function calculate the tSNR within the brain or spinal cord
        
        Attributes:
        ----------
        config: load config file
        ID: participant ID
        i_img: 4d func image
        inTemplate: put True to coregister the tSNR map into template space
        redo: put True to re-run the analysis on existing file (default=False)
    
    '''
    
    preproc_dir=config["main_dir"]+ config["preprocess_dir"]["bmpd_dir"] if ID[0]=="P" else config["main_dir"] +config["preprocess_dir"]["main_dir"]
    ID_preproc_dir=preproc_dir + "/sub-" + ID

    if i_img==None:
        i_img= glob.glob(ID_preproc_dir +"/func/"+config["preprocess_dir"]["func_moco"]  + structure + "/*_moco.nii.gz")[0]
        
    if mask==None:
        if structure=="spinalcord":
            mask=glob.glob(ID_preproc_dir +"/func/"+config["preprocess_dir"]["func_seg"]["spinalcord"] +"/*mean_seg.nii.gz")[0]
        else:
            mask_gmwm=glob.glob(ID_preproc_dir +"/anat/brain/segmentation/mri/p1p2*inFunc.nii*")[0]
            #mask_gmwmcsf=glob.glob(ID_preproc_dir +"/anat/brain/segmentation/mri/*mask.nii.gz")[0]
        

    native_tSNR= config["main_dir"]+config["tSNR_dir"] + "sub-"+ ID +"/" + os.path.basename(i_img).split(".")[0] + "_tSNR.nii"
    # compute tSNR *******************************************************************************
    if not os.path.exists(native_tSNR) or redo==True:
        print("coucou1")
        if not os.path.exists(os.path.dirname(native_tSNR)):
            os.mkdir(os.path.dirname(native_tSNR))
        tsnr_func= math_img('img.mean(axis=3) / img.std(axis=3)', img=i_img)
        tsnr_func_smooth = smooth_img(tsnr_func, fwhm=[3,3,6])
        tsnr_func_smooth.to_filename(native_tSNR)

    # extract value inside the mask
    o_txt=config["main_dir"]+config["tSNR_dir"] + "sub-"+ ID +"/sub-" + ID + "_" + structure +o_tag +"_tSNR_mean.txt"
    if not os.path.exists(o_txt) or redo==True:
        print("coucou2")
        if redo==True:
            os.remove(o_txt) 
        mask_name=["gmwm","gmwmcsf"]
        masks=[mask_gmwm] if structure=="spinalcord" else [mask_gmwm,mask_gmwmcsf]
        for mask_nb, mask in enumerate(masks):
            masker_stc = NiftiMasker(mask_img=mask,smoothing_fwhm=None,standardize=False,detrend=False) # select the mask
            tSNR_masked=masker_stc.fit_transform(native_tSNR) # mask the image
            mean_tSNR_masked=np.mean(tSNR_masked) # calculate the mean value
            

            with open(o_txt, 'a') as f:  # 'a' mode for appending to the file
                f.write(f"{mask_name[mask_nb]}: {mean_tSNR_masked}\n")  # Write the

            #with open(o_txt, 'w') as f:
                #f.write(str(mean_tSNR_masked))  # save in a file

        
    # Coregistratin into MNi or PAM50 space **************************************************
    if inTemplate == True:
        #>>>>>>>>>>>> Spinal cord
        project="stratals" if ID[0]=="A" else "bmpd"
        if structure=="spinalcord":
            preprocess_Sc=Preprocess_Sc(config)
            outreg_f=native_tSNR.split('.')[0] +"_inTemplate.nii.gz" #tSNR in template space
            if not os.path.exists(outreg_f):
                warp_img= glob.glob(ID_preproc_dir + config["warp_files"]["dir_sc"].format(ID,structure) + config["warp_files"]["warp_func2PAM50"])[0] # warping field
                preprocess_Sc.apply_warp(ID=ID,
                                         i_img=native_tSNR ,
                                         o_folder=os.path.dirname(native_tSNR)+ "/",
                                         dest_img=config["tools_dir"]["main_codes"] + "/template/"+config["PAM50_t2"],
                                         warping_field=warp_img,
                                         ses_name='',task_name='',tag='_inTemplate',
                                         threshold=None,redo=True,verbose=True) # apply transformation
            
        #>>>>>>>>>>>> Brain, in 2 steps
        if structure=="brain":
            preprocess_Br=Preprocess_Br(config)
           
            # >> Coreg into anat space
            coreg2anat_f=native_tSNR.split('.')[0] +"_inAnat.nii"
            if not os.path.exists(coreg2anat_f) or redo==True:
                # coregistration to anat space
                print(ID_preproc_dir + config["warp_files"]["dir_sc"].format(ID,structure) + config["warp_files"]["warp_func2PAM50"])
                
                anat_f= glob.glob(ID_preproc_dir + config["brain_anat"]["file"].format(ID))[0] 
                
                preprocess_Br.coregistration_func2anat(ID=ID,anat_img=anat_f,
                                              func_img=native_tSNR,
                                              tag="_inAnat",
                                              o_folder=os.path.dirname(native_tSNR) + "/")
                  
            # >> Coregitration into MNI space
            if not os.path.exists(coreg2anat_f.split('.')[0] +"_inTemplate.nii.gz"):
                warp_f=glob.glob(ID_preproc_dir  + config["warp_files"]["dir_brain"].format(ID,structure) + config["warp_files"]["warp_anat2MNI"])[0]
                
                outreg_f=coreg2anat_f.split('.')[0] + "_inTemplate.nii" # default output
                preprocess_Br.normalisation(ID=None,
                                            warp_file=warp_f,
                                            coreg2anat_file=coreg2anat_f,
                                            o_file=outreg_f,
                                            brain_mask=None,
                                            redo=False)
            outreg_f=coreg2anat_f.split('.')[0] + "_inTemplate.nii.gz" #tSNR in template space
            
    return o_txt, native_tSNR, outreg_f if inTemplate == True else None

def tSNR_group(config=None,i_img=None,structure='brain',o_tag="_moco",redo=False):
    
    if i_img==None:
        raise Warning("Provide a list of filenames !")

    # Create 4D image
    o_4d_img=config["main_dir"]+config["tSNR_dir"] + "/group/" + "4d_n" + str(len(i_img)) +  "_"+structure+"_tSNR.nii.gz" # output filename
    os.makedirs(os.path.dirname(o_4d_img), exist_ok=True) # create directory if not exists
    all_files=(' ').join(i_img) # join strings
    
    if not os.path.exists(o_4d_img) or redo==True:
        string="fslmerge -t " + o_4d_img + " " + all_files
        os.system(string)

    # Calculate mean image
    o_mean_img=config["main_dir"]+config["tSNR_dir"] + "/group/" + "mean_n" + str(len(i_img)) +  "_"+structure+"_tSNR.nii.gz" # output filename
    if not os.path.exists(o_mean_img) or redo==True:
        string="fslmaths " + o_4d_img + " -Tmean " + o_mean_img
        os.system(string)

    return o_mean_img

             
            
          
    
    
def plot_metrics(config,df=None,y=None,hue=None,index='ID',columns=['structure'],y_title="y_axis",save_plot=False):
        
        '''
        This function will help to plot different metrics
        
        Attributes:
        ----------
        df: dataframe with metrics informations
        y: values of the metrics to plot
        index: index column for each individuals
        columns: columns to plot as separate variables
            
        '''
                            
        # Make sure to remove the 'facecolor': 'w' property here, otherwise
        # the palette gets overrided
        colors=['#20b5bf','#ebb80b']

        # Make sure to remove the 'facecolor': 'w' property here, otherwise
        # the palette gets overrided
        boxprops = { 'linewidth': 2,'alpha':0.5}
        medianprops = {'linewidth': 2,'alpha':0.5}
        whiskerprops = {'linewidth': 1,'alpha':0.5}


        boxplot_kwargs = {'boxprops': boxprops, 'medianprops': medianprops,
                        'whiskerprops': whiskerprops, 'capprops': whiskerprops,
                        'width': 0.75, 'palette': colors}

        #stripplot_kwargs = { 'size': 6, 'alpha': 0.7,
        #                   'palette': pal, 'hue_order': hue_order}

        fig, ax = plt.subplots()

        sns.boxplot(data=df, x=columns, y=y,hue=hue, ax=ax,fliersize=0, **boxplot_kwargs)
        sns.stripplot(data=df, x=columns, y=y, hue=hue, ax=ax,palette=colors,
            dodge=True, jitter=0.2, )
        ax.legend_.remove()
             
        # Add axis ticks & labels
        #_ = ax.set_xticks(range(len(columns)))
        #_ = ax.set_xticklabels(columns2list)
        #_ = ax.set_ylabel(y_title)
        #plt.ylim(0,0.3)

        if save_plot==True:
            plot_f=config["tSNR"]["group_results_dir"] + "/mean_brsc_n" + str(len(config["participants_IDs"])) + "_" + y + ".pdf"
            plt.show()
            fig.savefig(plot_f,dpi=300, bbox_inches='tight')
    
