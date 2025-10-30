# -*- coding: utf-8 -*-
import glob, os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

class post_Cat12:
    '''
    The post_Cat12 class is used to extract results generate by cat12 toolbox such as brain gm and wm volume within ROIs
    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, config, IDs=None):
        '''
        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters
        contrast : string
            A string containing the contrast to extract could be MTR or T1w or T2s or dwi
        '''

        #1. Load configuration
        if config==None:
            raise Warning("Please provide the filename of the config file (.json)")
        self.config = config # load config info
        self.config["project_dir"]=config["project_dir"]
        if IDs==None:
            self.IDs=self.config["participants_IDs_ALL"]
        else :
            self.IDs==ID
        
        print("Your are going to run the analysis here:")
        
        participant_info_file=self.config["project_dir"] +config["population_infos"]
        self.metadata = pd.read_csv(participant_info_file,delimiter="\t")
        
        #2. Create the output directories
        print(self.config["project_dir"], self.config["analysis_dir"]["cat12"])
        self.outputdir = self.config["project_dir"] + self.config["analysis_dir"]["cat12"]
        self.firstlevel_dir = self.outputdir+ self.config["first_level"]
        self.secondlevel_dir = self.outputdir+ self.config["second_level"]


        os.makedirs(self.outputdir, exist_ok=True) # Create directories only if they do not exist
        os.makedirs(self.firstlevel_dir, exist_ok=True) # Create directories only if they do not exist
        os.makedirs(self.secondlevel_dir, exist_ok=True) # Create directories only if they do not exist

        
        #3. Individual directories
        self.cat12_dir=[]
        for ID in self.IDs:
            preproc_dir= self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
            self.cat12_dir.append(preproc_dir + self.config["cat12"]["dir"].format(ID))
            

    def read_catROI(self,IDs=None,i_tag="*_T1w_brain",atlas_name="Schaefer2018_200Parcels_7Networks_order",data=["Vgm","Vwm"],o_tag="",o_file=None,participant_info_file=None,redo=False,verbose=False,n_jobs=0):
        
        '''
        Read brain volume information  within .xlm output file
        
        Inputs
        ----------
        IDs: list
            list of the participants to process (e.g. ["A001","A002"])
        i_tag: string
            tagname of the input file
        atlas_name: string
            atlas name of the extracted rois, default: "Schaefer2018_200Parcels_17Networks_order"
        data: string
            name of the metric to extract default: ["Vgm","Vwm"]
        
             
        Returns
        ----------
          
        '''

        #----------- 1. check the inputs info 
        if IDs==None:
            IDs=self.IDs
        else:
            self.cat12_dir=[]
            for ID_nb, ID in enumerate(IDs):
                preproc_dir= self.config["preprocess_dir"]["bmpd"] if ID[0]=="P" else self.config["preprocess_dir"]["stratals"]
                self.cat12_dir.append(preproc_dir + self.config["cat12"]["dir"].format(ID))

        #----------- 2. Create the dataframe 
        catROI_f=[];all_data=[];
        for ID_nb, ID in enumerate(IDs):
            #-------- Extract info from xml file
            catROI_f.append(glob.glob(f"{self.cat12_dir[ID_nb]}/label/catROI_sub-{ID}{i_tag}.xml")[0])
            
            tree = ET.parse(catROI_f[ID_nb])  
            
            root = tree.getroot()
            rois = root.find(atlas_name)
            ids_text = rois.find("ids").text
            rois_ids = [int(x.strip()) for x in ids_text.strip("[]").split(";")]
            names_items = rois.find("names").findall("item")
            rois_names = [item.text for item in names_items]
            
            networks=[]; laterality=[];main_atlas=[]
            for network in rois_names:
                networks.append(network.split('_')[0][1:])
                lat="right" if network[0]=="r" else "left"
                laterality.append(lat)
                main_atlas.append(o_tag[1:])
  
            #-------- Get participant metadata
            participant_row = self.metadata[self.metadata['participant_id'] == ID]  # Extract the corresponding participant info
            output_f=[]
            if not participant_row.empty:
                output_f=self.firstlevel_dir + f"/sub-{ID}{i_tag[1:]}{o_tag}.csv"
                if not os.path.exists(output_f) or redo:
                    IDs = [] ;  groups = [] ; ages=[]; sex=[]
                    for i in range(0,len(rois_ids)):
                        IDs.append(ID)
                        groups.append(self.metadata [self.metadata ["participant_id"] == ID]["group"].values[0])
                        ages.append(self.metadata [self.metadata ["participant_id"] == ID]["age"].values[0])
                        sex.append(self.metadata [self.metadata ["participant_id"] == ID]["sex"].values[0])

                    #------ Extract values
                    data_text=[];metric_values=[]
                    for metric_nb,metric in enumerate(data):
                        data_text.append(rois.find("data").find(metric).text)
                        metric_values.append([float(x.strip()) for x in data_text[metric_nb].strip("[]").split(";")])
                        

                        # Build DataFrame
                        if metric_nb==0:
                            df = pd.DataFrame({
                                "IDs":IDs,
                                "age":ages,
                                "sex":sex,
                                "groups":groups,
                                "rois_nb": rois_ids,
                                "rois_name": rois_names,
                                "atlas":main_atlas,
                                "networks":networks,
                                "laterality":laterality,
                            metric: metric_values[metric_nb]})
                        else:
                            df[metric]=metric_values[metric_nb]

                    #----- save individual file
                    df.to_csv(output_f,index=False)
                
                else:
                    df=pd.read_csv(output_f)

                all_data.append(df)
        
        #----------- 3. Create a group level dataframe
        output_all_f=self.secondlevel_dir + str(len(self.IDs)) + i_tag +o_tag+".csv"
        all_df = pd.concat(all_data, ignore_index=True)
        if not os.path.exists(output_all_f) or redo:
            all_df.to_csv(output_all_f, index=False)
              

        return all_df



    