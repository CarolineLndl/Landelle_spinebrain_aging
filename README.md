# Spinal cord structural-functional architecture and its age related changes

**Cite:**
> TBD

**Preprint:**
>  Landelle C, Kinany K, St-Onge S, Lungu O, Van De Ville D, Misic B, Marchand-Pauvert V, De Leener B, Doyon J. 
Spinal cord structural and functional architecture and its shared organization with the brain across the adult lifespan 
bioRxiv 2025.10.02.679488;  doi: https://doi.org/10.1101/2025.10.02.679488

**Data:**
Functional data of the younger participant is available [here](https://openneuro.org/datasets/ds005075/).  
For other participants, please contact the authors.  

---
### Overview
This repository support the project on spinal cord structural-functional architecture and its age related changes.   
Most of the code was written in Python 3.10 and some analyses also were done in Matlab 2024b to access SPM.  
Toolboxes used include:
- Spinal Cord Toolbox (SCT, version 5.6.0; De Leener et al., 2017)  
- Oxford Center for fMRI of the Software Library (FSL, version 5.0)   
- Statistical Parametric Mapping (SPM12, running on Matlab 2021b)  
- Tapas PhysiO toolbox (release 2022a, V8.1.0; Kasper et al., 2017)  
- Nilearn toolbox (version 0.9.1)

<div style="background-color: #f2ebccff; padding: 10px;">
/!\ The upload of this repository is in progress    
- Preprocessing func (TBD)
- Denoising (TBD)
- Preprocessing anat (TBD)
- Analyses func (TBD)
- Analyses anat (TBD)
</div>

---  
### Repository  
The repository contains the following folders:  
- /code/
- /config/
- /notebook/
- /template/
  
    
#### <span style="background-color:#F0E8E6">/* </span>
- *CL_brsc_aging_env.sh* : project's environment


#### <span style="background-color:#F0E8E6">/code/ </span>
The code folder contain the different function used to run the analyses in the different notebook.  
You will see description in each script.  
- */code/spm/* : this folder contain the matlab scripts used with SPM. 

#### <span style="background-color:#F0E8E6">/config/ </span>
The config folder contain the different config files used in each notebook (.json)
- *participants_brsc_aging.tsv* : containe demographical information about the population


#### <span style="background-color:#F0E8E6">/notebook/ </span>
The notebook folder contain the different notebooks used to run and vizualized the results.  
Notebook's number correspond to the related figure number

#### <span style="background-color:#F0E8E6">/template/ </span>
Different images used in the analyses.
