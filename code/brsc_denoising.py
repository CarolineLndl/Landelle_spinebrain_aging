# Main imports ------------------------------------------------------------
import os, json, glob, shutil
from scipy.io import loadmat
import numpy as np
import pandas as pd
import nibabel as nib
import brsc_utils as utils

from joblib import Parallel, delayed


# Plotting
import matplotlib.gridspec as GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

#Stats
import scipy.stats as stats
from nipype.algorithms.confounds import compute_noise_components

#Nilearn
from nilearn import image


class Denoising:
    '''
    The Preproc class is used to run preprocessing analyses
    Attributes
    ----------
    config : dict
        '''
    def __init__(self,config):
        self.config = config # load config info
        # Create main dir
        if not os.path.exists(config["denoising"]["dir"].split("sub")[0]):
            os.makedirs(config["denoising"]["dir"].split("sub")[0])
        
        if not os.path.exists(config["normalization"]["dir"].split("sub")[0]):
            os.mkdir(config["normalization"]["dir"].split("sub")[0])
        
        if not os.path.exists(config["smoothing"]["dir"].split("sub")[0]):
            os.mkdir(config["smoothing"]["dir"].split("sub")[0])


        # Create individual directory if not exists _____________________________________________________________________________
        for run_nb, run in enumerate(config["design_exp"]["ses_names"]):
            for ID_nb,ID in enumerate(config["participants_IDs"]):

                # Create 1_func_denoised dir and individual dir
                den_dir= config["denoising"]["dir"].format(ID,run)
                print(den_dir)

                if not os.path.exists(den_dir):
                    os.makedirs(den_dir)

                if not os.path.exists(den_dir+'/'+ config["structures"][0]):
                    os.mkdir(den_dir +'/'+config["structures"][0]); os.mkdir(den_dir + '/'+config["structures"][0]+'/confounds/')
                if len(config["structures"]) ==2 and not os.path.exists(den_dir+'/'+ config["structures"][1]):
                    os.mkdir(den_dir +'/'+config["structures"][1]); os.mkdir(den_dir + '/'+config["structures"][1]+'/confounds/')
                        
                    print('Denoising folder created for ' + ID)

                # Create 2_func_in_template individual dir
                norm_dir=config["normalization"]["dir"].format(ID,run)

                if not os.path.exists(norm_dir):
                    os.makedirs(norm_dir)
                    
                if not os.path.exists(norm_dir +'/'+config["structures"][0]) :
                    os.mkdir(norm_dir+'/'+config["structures"][0])
                
                    if len(config["structures"]) ==2 and not os.path.exists(norm_dir+'/'+config["structures"][1]):
                        os.mkdir(norm_dir+'/'+config["structures"][1])
                

                    

                # Create 3_func_smoothed individual dir
                smooth_dir=config["smoothing"]["dir"].format(ID,run)

                if not os.path.exists(smooth_dir):
                    os.makedirs(smooth_dir)
                
                if not os.path.exists(smooth_dir+'/'+config["structures"][0]) :
                    os.mkdir(smooth_dir+'/'+config["structures"][0])
                if len(config["structures"]) ==2 and not os.path.exists(smooth_dir+'/'+config["structures"][1]):
                    os.mkdir(smooth_dir+'/'+config["structures"][1])
             



    def outliers(self,ID=None, input_file=None, structure=None,mask_file=None,run_name='', output_file=None,redo=False):
        '''
            Outliers calculation with fsl
            https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers

            Attributes
            ----------
            ID: participant ID (required , default=None)
            input_file : <filename> 4D func file, (required , default=None, moco file will be targeted)
            output_file : <filename> output files  (optional , default=None)
            mask_file : <filename> output files  (optional , default=None)                  
            outputs
           ----------
           to complete
                
        '''
        if ID==None:
            raise(Exception('ID shoul be provided ex: ID="A001"'))
        
        if structure==None:
            raise(Exception('structure should be provided ex: ID="spinalcord"'))
        
        # Define files:
        preproc_dir=self.config["preproc_dir_bmpd"] if ID[0]=="P" else self.config["preproc_dir"]
        if input_file==None:
            for run_nb, run in enumerate(self.config["design_exp"]["ses_names"]):
                print(preproc_dir + self.config["moco_files"]["dir"].format(ID,run,structure) + self.config["moco_files"]["moco_f"].format(ID,run,structure))
                input_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,run,structure) + self.config["moco_files"]["moco_f"].format(ID,run,structure))[0]

        if mask_file==None:
            for run_nb, run in enumerate(self.config["design_exp"]["ses_names"]):
                mask_file=glob.glob(preproc_dir + self.config["seg_files"][structure]["seg_func"].format(ID,run))[0]
        
        if output_file==None:
            for run_nb, run in enumerate(self.config["design_exp"]["ses_names"]):
                output_file=preproc_dir + self.config["outliers_files"]["dir"].format(ID,run,structure) + self.config["outliers_files"][ "outliers_f"].format(ID)

        
        if not os.path.exists(os.path.dirname(output_file)):


            os.makedirs(os.path.dirname(output_file))

        #get number of volumes
        
        
        # Run motion outliers calculation
        string="fsl_motion_outliers -i "+input_file+" -o "+output_file+ ".txt —m "+mask_file+ " --nomoco --dvars -p "+output_file + ".png "#-s " +output_file  +"_metric.txt"
        
        # fsl do not provide outputs if there are no outliers so we need to create a file with only 0 values
        if not os.path.exists(output_file +".txt") and redo==False and os.path.exists(output_file +".png"):
            func_img=nib.load(input_file)
            vol_number=func_img.header.get_data_shape()[3] 
            array = np.zeros((vol_number, 1))
            np.savetxt(output_file +".txt", array, fmt='%d', delimiter='   ')


        if not os.path.exists(output_file +".png") or redo==True:
            print(output_file +".txt")
            print("outliers calculation is running for participant " +ID + " " + structure)
            os.system(string)
     
        return output_file + ".txt"


    def find_physio_file(self,input_dir=None, ID=None, run_name='',copy=True,output_dir=None,redo=False):
        
        outputs=[];outputs_resp=[];outputs_puls=[];outputs_trig=[];outputs_tics=[]

        if input_dir==None:
            input_dir=self.config["raw_dir_bmpd"] + "/sub-" + ID + '/'+run_name + "/func/" if ID[0]=="P" else self.config["raw_dir"] + "/sub-" + ID + '/'+run_name + "/func/"
        

            
        # check if the physio_file exists

        physio_file_tsv=glob.glob(input_dir + '/*'+ run_name +'*.tsv*')
        physio_file_log=glob.glob(input_dir + '/*'+ run_name +'*.log')
        physio_file_mat=glob.glob(input_dir + '/**/*'+ run_name + 'RS.mat')

        if len(physio_file_tsv) ==1:
            if copy==True: # copy the file in an other folder if copy==True
                if output_dir is None:
                    raise(Exception('output_dir should be list of directories'))
                if not os.path.exists(output_dir+os.path.basename(physio_file_tsv[0])) or redo ==True:
                    output=shutil.copyfile(physio_file_tsv[0], output_dir+ '/'+os.path.basename(physio_file_tsv[0])) # copy the file in an other folder
                    print('Physio file has been copy here : ' + output)
                else:
                    output=output_dir+os.path.basename(physio_file_tsv[0])
                    
                if output.split('.')[-1] == "gz": # unzip the file if it was in .gz format
                    output=utils.unzip_file(i_file=output,ext='.tsv',zip_file=False,redo=redo)
                    
            else:
                output=physio_file_tsv
                if output.split('.')[-1] == "gz": # unzip the file if it was in .gz format
                    output=utils.unzip_file(i_file=output,ext='.tsv',zip_file=False,redo=redo)
                outputs.append(output)

        elif len(physio_file_tsv) >1:
            if copy==True: # copy the file in an other folder if copy==True
                if output_dir is None:
                    raise(Exception('output_dir should be list of directories'))
                print(input_dir + '/*'+ run_name +'*respiratory*.tsv*')
                input_resp=glob.glob(input_dir + '/*'+ run_name +'*respiratory*.tsv*')[0]
                input_puls=glob.glob(input_dir + '/*'+ run_name +'*cardiac*.tsv*')[0]
                input_resp_json=glob.glob(input_dir + '/*'+ run_name +'*respiratory*.json')[0]
                input_puls_json=glob.glob(input_dir + '/*'+ run_name +'*cardiac*.json')[0]

                if not os.path.exists(output_dir+ '/'+os.path.basename(input_resp)) or redo ==True:
                    output_resp=shutil.copyfile(input_resp, output_dir+ '/'+os.path.basename(input_resp))
                    shutil.copyfile(input_resp_json, output_dir+ '/'+os.path.basename(input_resp_json))
                    output_puls=shutil.copyfile(input_puls, output_dir+ '/'+os.path.basename(input_puls))
                    shutil.copyfile(input_puls_json, output_dir+ '/'+os.path.basename(input_puls_json))
                    print('Physio file has been copy here : ' + output_resp)

                else:
                    output_resp=output_dir+os.path.basename(input_resp)
                    output_puls=output_dir+os.path.basename(input_puls)
                    
                if output_resp.split('.')[-1] == "gz": # unzip the file if it was in .gz format
                    output_resp=utils.unzip_file(i_file=output_resp,ext='.tsv',zip_file=False,redo=redo)
                    output_puls=utils.unzip_file(i_file=output_puls,ext='.tsv',zip_file=False,redo=redo)
                
                
        elif len(physio_file_log) >0:
            # multiple log files exists (e.g *_RESP.log; *_PULS.log; *_Trigger.log)
            input_resp=glob.glob(input_dir + '*'+ run_name +'*RESP.log')[0]
            input_puls=glob.glob(input_dir + '*'+ run_name +'*PULS.log')[0]
            #input_trig=glob.glob(inputs_dir[sbj_nb] + '*'+ run_name +'*Trigger*.log')[0]
            input_tics=glob.glob(input_dir + '*'+ run_name +'*AcquisitionInfo.log')[0]
            
            
            if copy==True: # copy the file in an other folder if copy==True
                if output_dir is None:
                        raise(Exception('output_dir should be list of directories'))
                
                if not os.path.exists(output_dir[sbj_nb]+os.path.basename(input_resp)) or redo ==True:    
                    output_resp=shutil.copyfile(input_resp, output_dir[sbj_nb]+ '/'+ os.path.basename(input_resp)) # copy the file in an other folder
                    output_puls=shutil.copyfile(input_puls, output_dir[sbj_nb]+ '/'+os.path.basename(input_puls))
                    #output_trig=shutil.copyfile(input_trig, outputs_dir[sbj_nb]+ '/'+os.path.basename(input_trig))
                    output_tics=shutil.copyfile(input_tics, output_dir[sbj_nb]+ '/'+os.path.basename(input_tics))
                else:
                    output_resp=output_dir[sbj_nb]+os.path.basename(input_resp) # copy the file in an other folder
                    output_puls=output_dir[sbj_nb]+os.path.basename(input_puls)
                    #output_trig=outputs_dir[sbj_nb]+os.path.basename(input_trig)
                    output_tics=output_dir[sbj_nb]+os.path.basename(input_tics)
            
            else:
                output_resp=input_resp;output_puls=input_puls;output_tics=input_tics
                #output_trig=input_trig;  
            output_resp.append(output_resp); output_puls.append(output_puls); output_tics.append(output_tics) #outputs_trig.append(output_trig) ; 
        
        elif len(physio_file_mat) >0:   
            if copy==True:# copy the file in an other folder if copy==True
                if outputs_dir is None:
                    raise(Exception('output_dir should be list of directories'))
                if not os.path.exists(physio_file_mat[0]) or redo ==True:
                    output=shutil.copyfile(physio_file_mat[0], output_dir+ '/'+os.path.basename(physio_file_mat[0])) # copy the file in an other folder
                    print('Physio file has been copy here : ' + output) 
                else:
                    output=output_dir+os.path.basename(physio_file_mat[0])

        else:
            raise(Exception('Physio files format should be in .log or .tsv or .tsv.gz or .mat'))
                
        return output if len(physio_file_tsv) == 1 or len(physio_file_mat) > 0 else (output_resp, output_puls) if len(physio_file_tsv) == 2 else (output_resp, output_puls, output_tics) if len(physio_file_log) > 0 else None

    
    def moco_params(self,ID=None, slice_wise=True,input_file=None, func_file=None, structure="spinalcord",run_name='', output_file=None,redo=False,verbose=True):
        '''
            Create slicewise moco parameters

            Attributes
            ----------
            input_file : list
                list of moco param file (you can provide 2 for the same participant)

            func_img: filename
                The brain functional image will be useful to extract the number of slices, it is not necessary for spinal cord as the information is already included in moco_param image.
            structure: "brain" or "spinal cord", default="brain"
            
            output_dir : list
                list of output files (one for each participants)
            save: Bolean
                optional, to save the plots put True (default: False)
                
        '''

        if ID==None:
            raise(Exception('ID should be provided ex: ID="A001"'))
        preproc_dir=self.config["preproc_dir_bmpd"] if ID[0]=="P" else self.config["preproc_dir"]
        PhysioDir=self.config["denoising"]["dir"].format(ID,run_name)  +'/'+ structure +'/confounds/' # output directory

        # Select the input file  (text file if brain or nifti file if spinal cord)
        if input_file==None: 
            if structure=="brain":
                input_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,run_name,structure) + self.config["moco_files"]["moco_param"][structure])[0]

            else:
                input_file=sorted(glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,run_name,structure) + self.config["moco_files"]["moco_param_img"][structure]))

        if slice_wise:
            if structure=="brain":
                if func_file==None:
                    func_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,run_name,structure) + self.config["moco_files"]["moco_mean_f"])[0] 
                
                func_img=nib.load(func_file) # load the func image
                moco_brain=pd.read_csv(input_file, delim_whitespace=True, header=None) # load motion parameter file
                for slice_nb in range(0,func_img.header.get_data_shape()[2]):
                    slice_str="00" + str(slice_nb + 1) if (slice_nb+1)<10 else "0" + str(slice_nb+1)
                    output_moco_file=PhysioDir +  '/sub-' + ID +  '_6_'+'moco_'+structure+'_slice'+slice_str+'.txt'
                    if not os.path.exists(output_moco_file):
                        np.savetxt(output_moco_file, moco_brain)

                if os.path.exists(output_moco_file) and redo==False and verbose==True:
                    print("Brain moco params were already extracted please, put redo=True to recalculate it")
                

                # create a dataframe with volume value for each slice as we do not have the slice wise motion corrected parameters
                
            if structure=="spinalcord":
                X_img=nib.load(input_file[0]) # load the X moco parameter image
                Y_img=nib.load(input_file[1]) # load the X moco parameter image
                
                #extract the mocovalue for each slice
                for slice_nb in range(0,X_img.header.get_data_shape()[2]):
                    slice_str="00" + str(slice_nb + 1) if (slice_nb+1)<10 else "0" + str(slice_nb+1)
                    output_moco_file=PhysioDir +  '/sub-' + ID +  '_2_'+'moco_'+structure+'_slice'+slice_str+'.txt'
                    if not os.path.exists(output_moco_file):
                        moco_value=[]
                        for img in [X_img,Y_img]:
                            img_slice=img.slicer[:,:,slice_nb:slice_nb+1,:] # cropped func slices
                            imgseries=img_slice.get_fdata(dtype=np.float32)
                            imgseries_reshape=imgseries.reshape(265, 1)

                            moco_value.append(imgseries_reshape)
                        moco_value=np.hstack(moco_value)
                    
                        np.savetxt(output_moco_file, moco_value)
                    
                if os.path.exists(output_moco_file) and redo==False and verbose==True:
                    print("Spinal cord moco params were already extracted please, put redo=True to recalculate it")
            
        else:
            # moco param are going to by copy
                
            input_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,structure) + self.config["moco_files"]["moco_param"][structure])[0]
            output_moco_file = os.path.join(
                PhysioDir,
                f"sub-{ID}_{'6' if structure == 'brain' else '2'}_moco_{structure}.txt")

            if not os.path.exists(output_moco_file):
                delimiter=" " if structure == 'brain' else ","
                moco_brain=pd.read_csv(input_file, delimiter=delimiter, header=None) # load motion parameter file
                np.savetxt(output_moco_file, moco_brain)


            
    def plot_physio(self,denoising_mat,output_dir,ID=None,save=False,verbose=False):
        '''
            Plot physiological recordings

            Attributes
            ----------
            denoising_mat : list
                list of 4D input files (one for each participants)
            output_dir : list
                list of output files (one for each participants)
            save: Bolean
                optional, to save the plots put True (default: False)
                
            outputs
           ----------
           to complete
                
        '''
        if save==True:
            # 1. Load denoising matrice
            mat_file={}
            load_mat= loadmat(denoising_mat)
            mat_file_load=load_mat['physio']['ons_secs'][0][0][0][0]
            mat_file={'t':mat_file_load[0],'t_start':mat_file_load[1],
                                'c':mat_file_load[2],'r':mat_file_load[3], # raw recordings across time c:cardiac and r: respi
                                'c_scaling':mat_file_load[4], 'r_scaling':mat_file_load[5], # 3 data / secondes
                                'c_is_reliable':mat_file_load[6],'r_is_reliable':mat_file_load[7],
                                'c_pulse':mat_file_load[8], # peaks / secondes
                                'fr':mat_file_load[9],
                                'c_sample_phase':mat_file_load[10],
                                'hr':mat_file_load[12], # heart rate / secondes
                                'rvt':mat_file_load[13]} # respiratory volume per time (per secondes)

        # 2. Convert .mat in .txt files_______________________________________
            np.savetxt(output_dir +  '/sub-' + ID +  '_raw_resp.txt', mat_file['r'])
            np.savetxt(output_dir +  '/sub-' + ID +  '_raw_card.txt', mat_file['c'])
            np.savetxt(output_dir +  '/sub-' + ID + '_hr.txt', mat_file['hr'])
            np.savetxt(output_dir +  '/sub-' + ID +  '_rvt.txt', mat_file['rvt'])

                # 3 Plots physio and save figures_________________________________________

            fig=plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
            gs=GridSpec.GridSpec(2,2,figure=fig) # 2 rows, 3 columns
            ax1=fig.add_subplot(gs[0,0]) # First row, first column
            ax2=fig.add_subplot(gs[0,1]) # First row, second column
            ax3=fig.add_subplot(gs[1,:]) # Second row, span all columns
            fig.tight_layout()


            fig.subplots_adjust(hspace = .5, wspace=0.1)
            ax3.stem((mat_file['c_pulse'][0:]),np.ones(len(mat_file['c_pulse'][0:])),'lightpink',markerfmt=' ', basefmt=' ' )

            ax3.plot((mat_file['t'][1:]),mat_file['c'][1:],color='crimson')
            ax1.plot(np.arange(start=0, stop=len(mat_file['c_sample_phase'])),(mat_file['c_sample_phase']+70) ,color='crimson')

            ax1.plot(mat_file['hr'],color='black')
            ax2.plot(np.arange(start=0, stop=len(mat_file['t']))/self.config["acq_params"]["physio_frq"][0]/self.config["acq_params"]["TR"],mat_file['r'])
            ax2.plot(mat_file['rvt'],color='black')
            ax1.set_title('sub-' + ID ,fontsize=20)

            ax3.set_ylabel("Card / Peak Cardio ",fontsize=12)
            ax1.set_ylabel("resample card / Hear rate ",fontsize=12)
            ax1.set_xlabel("Volumes",fontsize=12)
            ax2.set_ylabel("Respiration and  RVT",fontsize=12) # rvt= respiratory volume per time
            ax2.set_xlabel("Volumes",fontsize=12)
            ax3.set_xlabel("Time (sec)",fontsize=12)
            plt.savefig(output_dir   + '/sub-' + ID + '_Physio.png', dpi=300, bbox_inches = 'tight')
            plt.close()
            if verbose==True:
                print("physio plot were saved") 
                

        else:
            print("plot were not saved put save=True to save it")

    def confounds_ts(self,ID=None,func_file=None,slice_wise=True,mask_seg_file=None,mask_csf_file=None,compcor=False, DCT=False, output_file=None, run_name="", structure="brain", n_compcor=5, n_DCT=3, TR=None, redo=False, verbose=False):
        '''
          Generate aCompCor masks
          https://nipype.readthedocs.io/en/1.1.0/interfaces/generated/nipype.algorithms.confounds.html

            Attributes
            ----------
            ID str
                participant ID (required , default=None you should provide a str like "A001"
            
            func_file <filename>
                filename of the input 4D file (default=None)
            
            slice_wise <bool>
                Whether the metric are going to be extracted slice_wise (default) of volume_wise 
                True the metrics are extracted within each slice of the image (default)
                False the metric  are extracted within the entire image (e.g. entire CSF mask for CompCor)
            
            mask_seg_file <filename>
                filename of a binary mask of the gm or wm+gm, it will be use to compute the DCT (default=None)
            
            mask_csf_file <filename>
                filename of a binary mask of the csf, it will be use to compute the aCompCor (default=None)
                
            compcor <bool>
                Put True to run aCompCor (default=False)

            DCT <bool>
                Put True to run DCT (default=False)

            output_file=None
            run_name=""
            structure="brain"
            n_compcor=5
            n_DCT=3,
            TR=None
            redo=False
            verbose=False
            
                
        '''
        if ID==None:
            raise(Exception('ID shoul be provided ex: ID="A001"'))

        if func_file==None:
            func_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,structure) + self.config["moco_files"]["moco_f"].format(ID,structure))[0]

        if mask_seg_file==None:
             mask_seg_file=glob.glob(preproc_dir + self.config["seg_files"][structure]["dir_func"].format(ID) + self.config["seg_files"][structure]["seg_func"])[0]
        
        if mask_csf_file==None:
             mask_csf_file=glob.glob(preproc_dir + self.config["seg_files"][structure]["dir_func"].format(ID) + self.config["seg_files"][structure]["csf_infunc"])[0]

        if TR== None:
            TR=self.config["acq_params"]["TR"]
            
        func_img=nib.load(func_file) # load the functional image
        mask_seg_img=nib.load(mask_seg_file) # load the seg mask image
        mask_csf_img=nib.load(mask_csf_file) # load the csf mask image


        PhysioDir=self.config["denoising"]["dir"].format(ID,run_name) +'/'+ structure +'/confounds/' # output directory
            
        # used the lower number of slices to extract noise
        n_slices = min(func_img.header.get_data_shape()[2], 
               mask_seg_img.header.get_data_shape()[2], 
               mask_csf_img.header.get_data_shape()[2])

        if slice_wise:
            output_DCT_file=PhysioDir +  '/sub-' + ID +  '_'+str(n_DCT)+'_DCT_'+structure+'_slice001.txt'
            output_compcor_file=PhysioDir +  '/sub-' + ID +  '_'+str(n_compcor)+'_acompcor_'+structure+'_slice001.txt'
        else:
            output_DCT_file=PhysioDir +  '/sub-' + ID +  '_'+str(n_DCT)+'_DCT_'+structure+'.txt'
            output_compcor_file=PhysioDir +  '/sub-' + ID +  '_'+str(n_compcor)+'_acompcor_'+structure+'.txt'

        if not os.path.exists(output_compcor_file) or not os.path.exists(output_DCT_file) or redo:
            if slice_wise: 
                # loop over the different slices
                for slice_nb in range(0,n_slices):
                    func_slice=func_img.slicer[:,:,slice_nb:slice_nb+1,:] # cropped func slices
                    mask_seg_slice=mask_seg_img.slicer[:,:,slice_nb:slice_nb+1] # cropped mask slices
                    mask_csf_slice=mask_csf_img.slicer[:,:,slice_nb:slice_nb+1] # cropped mask slices
                    slice_str="00" + str(slice_nb + 1) if (slice_nb+1)<10 else "0" + str(slice_nb+1)

                    # Run DCT
                    if DCT:
                        output_DCT_file=PhysioDir +  '/sub-' + ID +  '_'+str(n_DCT)+'_DCT_'+structure+'_slice'+slice_str+'.txt'
            

                        if not os.path.exists(output_DCT_file) or redo:
                            DCT_comp=compute_noise_components(imgseries=func_slice.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_seg_slice],
                                                            filter_type='cosine', # 'cosine': Discrete cosine (DCT) basis
                                                            period_cut=128,  # 'period_cut': minimum period (in sec) for DCT high-pass filter
                                                            repetition_time=TR,
                                                            components_criterion=n_DCT)
                                            
                            if DCT_comp[0].size==0:
                                DCT_comp_final=np.full((func_img.header.get_data_shape()[3],n_DCT), np.nan)

                            else:
                                DCT_comp_final=DCT_comp[1]

                            np.savetxt(output_DCT_file, DCT_comp_final)

                    # Run compcor
                    if compcor:
                        output_compcor_file=PhysioDir +  '/sub-' + ID +  '_'+str(n_compcor)+'_acompcor_'+structure+'_slice'+slice_str+'.txt'
                        ##############
                        
                        if not os.path.exists(output_compcor_file) or redo:
                            compcor_comp_final=[]
                            
                            compcor_comp=compute_noise_components(imgseries=func_slice.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_csf_slice], filter_type='polynomial', degree=2,
                                                            repetition_time=TR,
                                                            components_criterion=n_compcor)
                            
                            # create a matri with 0 value when there is no mask, it can happen of extrem slices
                            # A zero predictor contains no variability or information, so it cannot influence the signal during regression.
                            if compcor_comp[0].size==0:
                                compcor_comp_final=np.full((func_img.header.get_data_shape()[3],n_compcor), 0)

                            else:
                                compcor_comp_final=compcor_comp[0]
                            np.savetxt(output_compcor_file, compcor_comp_final)
                
            else: 
                #extract the metric across the volume
                #Run DCT
                if DCT:
                    if not os.path.exists(output_DCT_file) or redo:
                        DCT_comp=compute_noise_components(imgseries=func_img.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_seg_img],
                                                            filter_type='cosine',# 'cosine': Discrete cosine (DCT) basis
                                                            period_cut=128, # 'period_cut': minimum period (in sec) for DCT high-pass filter
                                                            repetition_time=TR,
                                                            components_criterion=n_DCT)
                        np.savetxt(output_DCT_file,DCT_comp[1])

                # Run compcor
                if compcor:                   
                    if not os.path.exists(output_compcor_file) or redo:
                        compcor_comp=compute_noise_components(imgseries=func_img.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_csf_img],
                                                            filter_type='polynomial', #'polynomial' Legendre polynomial basis
                                                            degree=2,
                                                            repetition_time=TR,
                                                            components_criterion=n_compcor)
                            
                        np.savetxt(output_compcor_file, compcor_comp[0])
        
            
        return output_compcor_file, output_DCT_file
   

    
    def combine_confounds(self,ID=None,confounds_infos=None,func_file=None,structure=None,retroicor_confounds=False,compcor_confounds=False,moco_confounds=False,outliers_confounds=False,DCT_confounds=False,slice_wise=True,run_name="",redo=False):
        '''
            Combine confounds in a same file 
            to do: improve the reading of the motion parameters (i.e different delimiters with fsl and sct)
            
            Attributes
            ----------
            
            confounds_infos: dict (required , default=None)
                should have the following form: {'Outliers':0,'Motion':2,'Retroicor':18,'CompCor':5,'DCT:3'} if one of the variable will not be use remove it from the dictionnary (don't put a 0)
            ID: participant ID (required , default=None
            func_files : <filename> 4D func file, (required , default=None, moco file will be targeted)
            structure : 'brain' or 'spinalcord' name of the structure to work on (requiered, default=None)         
            tapas_confounds : <filename> .txt physio confounds file (optional,default=False)            
            motion_param: <filename>  moco parameters file (optional, default:False)         
            outliers: <filename> .txt outliers confounds files (default: None)            
            DCT: .txt DCT files (default=False)            
            save: Bolean
                optional, to save the plots put True (default: False)
                
            return
            ----------
            output_file: .txt output file 
            output_zfile: .txt output files after z-scoring 
            Counfounds: confounds in dictionnary format
                
        '''
        if ID==None:
            raise(Exception('ID shoul be provided ex: ID="A001"'))

        if confounds_infos==None:
            raise(Exception("Provide confound info: ex: {'Outliers':0,'Motion':6,'Retroicor':18,'CompCor':12,'DCT':3}"))

        # select optional files:
        preproc_dir=self.config["preproc_dir_bmpd"] if ID[0]=="P" else self.config["preproc_dir"]
        if func_file==None:
            
            func_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,run_name,structure) + self.config["moco_files"]["moco_f"].format(ID,structure))[0]

        if "Outliers" in confounds_infos:
            outliers_read=pd.read_csv(outliers_confounds,sep='  ',index_col=False,header=None,engine='python')
            outliers_sbj=pd.DataFrame.to_numpy(outliers_read)
        

        func_img=nib.load(func_file) # load the func image
        slice_number=func_img.header.get_data_shape()[2] if slice_wise==True else 1
        slice_range = range(slice_number) if slice_wise else [None]

        for slice_nb in slice_range:

            slice_str = f"{slice_nb + 1:03d}" if slice_wise else ""
            output_tag = f"_slice{slice_str}" if slice_wise else ""
            run_tag="_" + run_name if run_name != "" else ""
                
            output_file = os.path.join(self.config["denoising"]["dir"].format(ID,run_name), structure, "confounds", f"sub-{ID}_allconfounds_{structure}{run_tag}{output_tag}.txt")

            
            if not os.path.exists(output_file) or redo:
                confounds_file = {}
                Confounds = {'All': np.empty((0, image.load_img(func_file).shape[3]))}
                
                for confound_name, n_comp in confounds_infos.items():
                     #print(confound_name)
                    confound_path = None
                    if confound_name != "Outliers":
                        confound_path = glob.glob(os.path.join(self.config["denoising"]["dir"].format(ID,run_name), structure, "confounds", f"{self.config['denoising'][confound_name + '_file']}*{slice_str}.txt"))
                    
                    if confound_path:  # Checks if the list is non-empty
                        confounds_file[confound_name] = confound_path[0]
                        Confound_read = pd.read_csv(confounds_file[confound_name], sep='\s+', index_col=False, header=None)
                        Confound_df = Confound_read.to_numpy()
                        Confounds[confound_name] = Confound_df[:, :n_comp]
                    else: # if empty put nan
                        Confounds[confound_name] = np.full((func_img.header.get_data_shape()[3], n_comp), np.nan)
                    
                    if n_comp == 0:
                        raise Exception('Number of component confound should be higher than 0 or you should set None to the confound name')
                    
                    #print(Confounds[confound_name].shape)
                    
                    #if confound_name=='HR' and retroicoir_confounds is not None:
                        #Confounds['HR']=retroicor_read[:,n_Retroicor:n_Retroicor+1]
                    
                    #if confound_name=='RVT' and retroicor_confounds is not None:
                        #Confounds['RVT']=retroicor_read[:,n_Retroicor+1:n_Retroicor+2]
                        
                    if confound_name == 'Outliers' and outliers_confounds is not None:
                        Confounds['Outliers'] = outliers_sbj
                    
                    Confounds['All'] = np.concatenate((Confounds['All'], Confounds[confound_name].T))
                
                Counfounds_dataframe = pd.DataFrame(Confounds['All'].T)
                Confounds_zscores = stats.zscore(Confounds['All'].T)
                Confounds_zscores = np.nan_to_num(Confounds_zscores, nan=0.0, posinf=0.0, neginf=0.0)
                Confounds_zscores_dataframe = pd.DataFrame(Confounds_zscores)
                Counfounds_dataframe.to_csv(output_file, index=False, header=False, sep=' ')
                Confounds_zscores_dataframe.to_csv(f"{output_file.split('.')[0]}_z.txt", index=False, header=False, sep=' ')

            output_zfile=output_file.split('.')[0] + '_z.txt'

        return output_file, output_zfile




    def plot_confound_design(self,ID=None,confound_file=None,confounds_infos=None,structure=None,run_name='',save=False):
        '''
            Plot confound design matrix

            Attributes
            ----------
            confound_files : list
                list of 4D input files (one for each participants)
            confounds_infos : dict
                dictonary with the name an the number of each confounds: (e.g {'Outliers':0,'Motion':6,'Retroicor':18,'CompCor':5})
            save: Bolean
                optional, to save the plots put True (default: False)
                       
        '''
        if ID==None:
            raise(Exception('ID should be provided ex: ID="A001"'))
        
        if structure==None:
            raise(Exception('Structure should be provided ex: structure="spinalcord"'))

        Confounds=pd.read_csv(confound_file,delimiter=' ',index_col=False,header=None)
        total_confounds=0
            
        for confound_name in confounds_infos:
            total_confounds=total_confounds+confounds_infos[confound_name]
                
                
        for confound_name in confounds_infos:
            if confound_name == "Outliers":
                confounds_infos["Outliers"]=Confounds.shape[1]+1-total_confounds
            elif confound_name =="Outliers_brsc":
                confounds_infos["Outliers_brsc"]=Confounds.shape[1]+1-total_confounds       
            
        labels=['']
        for confound_name in confounds_infos:
            labels=np.concatenate((labels,np.repeat(confound_name,confounds_infos[confound_name])))
        
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax=sns.heatmap(Confounds[:],vmin=-1, vmax=1,xticklabels=labels[1:])# change subject name to check an other subject
        ax.set_title('Confound Matrix ' +structure+ ' ' +run_name +' participant: '  + ID,fontsize = 15) 
        ax.set_ylabel('Volumes',fontsize = 12)
        ax.set_xlabel('Confounds',fontsize = 12)
            
            
        if save==True:
            plt.savefig(confound_file.split('.')[0]+'.png')
        
       # confounds_infos["Outliers"]=0

           
    def clean_images(self,ID=None,slice_wise=True,func_file=None,structure="brain",output_file=None,confounds_file=None,mask_file=None,run_name='',standardize="zscore",detrend=False,high_pass=0.01,low_pass=0.17,tag_name="",n_jobs=1,redo=False):
        
        ########### Check initiation:
        if ID==None:
            raise(Exception('ID should be provided ex: ID="A001"'))

        if structure==None or confounds_file == None :
            raise(Exception("'structure', 'confounds_files' and 'confound_infos' are required "))
        
        ###########  Load the func file and mask to extract the number of slices and the TR:
        preproc_dir=self.config["preproc_dir_bmpd"] if ID[0]=="P" else self.config["preproc_dir"]
        if func_file==None:
            func_file=glob.glob(preproc_dir + self.config["moco_files"]["dir"].format(ID,run_name,structure) + self.config["moco_files"]["moco_f"].format(ID,structure))[0]

        func_img=nib.load(func_file) # load the func image
        

        slice_number=func_img.header.get_data_shape()[2] if slice_wise==True else 1 # extract the number of slices
        TR=func_img.header.get_zooms()[3] # extract TR value


        ########### Run the loop for each slice:
        output_dir=self.config["denoising"]["dir"].format(ID,run_name) + "/" + structure + "/tmp/" if slice_wise else self.config["denoising"]["dir"].format(ID) + "/" + structure +"/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir) # create a temporary folder for slicewise denoising
        
        output_main_file= self.config["denoising"]["dir"].format(ID,run_name) + "/" + structure +"/" +os.path.basename(func_file.split('.')[0] + "_"+tag_name+ '.nii.gz')

        if not os.path.exists(output_main_file) or redo==True:
            
            slice_number=func_img.header.get_data_shape()[2] if slice_wise==True else 1
            slice_range = range(slice_number) if slice_wise else [None]

            for slice_nb in range(0,slice_number):
                slice_str = f"{slice_nb + 1:03d}" if slice_wise else ""
                confounds_f_slice = confounds_file.split("_slice")[0] +"_slice" + str(slice_str) + ".txt" if slice_wise else confounds_file
                output_tag="_slice"+str(slice_str) if slice_wise==True else ""
                output_file=output_dir +os.path.basename(func_file.split('.')[0] + "_"+tag_name+ output_tag+'.nii.gz')
                
                if not os.path.exists(output_file) or redo==True:
                    mask_img=nib.load(mask_file) # load the mask image
                    if slice_wise:
                        func_slice=func_img.slicer[:,:,slice_nb:slice_nb+1,:] # cropped func slices
                        
                        mask_slice=mask_img.slicer[:,:,slice_nb:slice_nb+1] # cropped mask slices
                    else:
                        func_slice=func_img
                        mask_slice=mask_img

                    
                    # extract the mask value to check if there are not empty if so do not denoised this slice
                    data = mask_slice.get_fdata()
                    
                    if np.mean(data) != 0:
                        Clean_image=image.clean_img(func_slice, 
                                                confounds= confounds_f_slice,
                                                mask_img=mask_slice, 
                                                detrend=detrend,
                                                standardize=standardize,
                                                low_pass=low_pass, 
                                                high_pass=high_pass,
                                                t_r=TR)

                        Clean_image.to_filename(output_file) #save image
                    else:
                        func_slice.to_filename(output_file)

        
            if slice_wise:
                # merge each slices in a single img
  
                if not os.path.exists(output_main_file) or redo:
                    nifti_files = glob.glob(os.path.join(self.config["denoising"]["dir"].format(ID,run_name) + "/" + structure +"/tmp/", "*.nii.gz"))
                    nifti_files.sort()  # Alphabetical sort 
                    
                    fsl_command="fslmerge -z " + output_main_file + " " + " ".join(nifti_files)
                    os.system(fsl_command)# run fsl command

                    shutil.rmtree(self.config["denoising"]["dir"].format(ID,run_name) + "/" + structure +"/tmp/") #remove the tmp folder

        
        output_meanfinal_file=output_main_file.split(".")[0] + "_mean.nii.gz"

        if not os.path.exists(output_meanfinal_file):
            fsl_command="fslmaths " + output_main_file + " -Tmean " + output_meanfinal_file
            os.system(fsl_command)# run fsl command
        
        return output_main_file


    def standardize(self,input_files,output_files,json_files=None,mask_files=None,redo=False):
        #demean and standardized the signal by the std
        if not os.path.exists(input_files[0]) or redo==True:
            for file_nb in range(0,len(input_files)):
                timeseries=nib.load(input_files[file_nb]).get_fdata() # extract Time series dats
                signals= timeseries.reshape(-1, timeseries.shape[-1]).T # reshape timeseries (nb_volumes, nb_voxels)

                if signals.shape[0] == 1:
                    warnings.warn('Standardization of 3D signal has been requested but '
                              'would lead to zero values. Skipping.')
                else:
                    signals= timeseries.reshape(-1, timeseries.shape[-1]).T # reshape timeseries (nb_volumes, nb_voxels)
                    mean =  signals.mean(axis=0)
                    std = signals.std(axis=0)
                    std[std < np.finfo(np.float64).eps] = 1.  # avoid numerical problems
                    signals=signals-mean # demean
                    signals/= std

                # save into filename
                std_image=image.new_img_like(input_files[file_nb], signals.T.reshape(timeseries.shape),copy_header=True)
                std_image.to_filename(output_files[file_nb]) #save image

                if mask_files:
                    string='fslmaths ' + output_files[file_nb] + ' -mas ' + mask_files[file_nb] + ' ' + output_files[file_nb] 
                    os.system(string)

                if json_files is not None:
                    infos={"standardize":True,
                          "mask":mask_files[sbj_nb]}
                    with open(json_files[file_nb], 'w') as f:
                        json.dump(infos, f) # save info
                    
