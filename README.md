# Spinal cord structural-functional architecture and its shared organization with the brain across the adult lifespan 
**Cite:**
> TBD

**📜 Preprint:**
>  Landelle C, Kinany K, St-Onge S, Lungu O, Van De Ville D, Misic B, Marchand-Pauvert V, De Leener B, Doyon J. 
Spinal cord structural and functional architecture and its shared organization with the brain across the adult lifespan 
bioRxiv 2025.10.02.679488;  doi: https://doi.org/10.1101/2025.10.02.679488

**💾 Data:**  
Functional data of the younger participant is available [here](https://openneuro.org/datasets/ds005075/).  
Derivatives data (preprocessed and denoised) from a single participant can be found here (A006):
For other participants, please contact the authors.  

---
### Overview
This repository support the project on spinal cord structural-functional architecture and its age related changes.   
Most of the code was written in Python 3.10 and some analyses also were done in Matlab 2024b to access SPM.  

🛠️ Toolboxes used include:
- Spinal Cord Toolbox (SCT, version 5.6.0; De Leener et al., 2017)  
- Oxford Center for fMRI of the Software Library (FSL, version 5.0)   
- Statistical Parametric Mapping (SPM12, running on Matlab 2021b)  
- Tapas PhysiO toolbox (release 2022a, V8.1.0; Kasper et al., 2017)  
- Nilearn toolbox (version 0.9.1)

<div style="background-color: #f2ebccff; padding: 10px;">

- Preprocessing func, microstructural and diffusion ✅   </br>
- Func Denoising ✅    </br>
- Fig 1, S6 ✅    </br>
- Fig 2 SpiDyn, SpiFC, coupling ✅ </br>
- Fig 3 morpho-SpiFC, morpho-SpiDyn ✅</br>
- Fig 4, S7 Brain/Spinal morpho   ✅, FC ✅, Dyn ✅</br>
- Fig S2 QC ✅</br>
- Fig S3 iCAPs ✅</br>
- Fig S5 Tract-specific analyses ✅</br>
- Fig S8 Distribution of the SpiDyn features ✅</br> </div>

---  
### Repository  
The repository contains the following folders:  
- /code/
- /config/
- /notebook/
- /template/
  



#### <span style="background-color:#F0E8E6">/notebook/ </span>
The notebook folder contain the different notebooks used to run and vizualized the results.  
Notebook's number correspond to the related figure number
- /notebook/preprocessing/ : notebooks used for preprocessing

#### <span style="background-color:#F0E8E6">/template/ </span>
Different images used in the analyses.
- *PAM50* : for spinal cord preprocessing
- *MNI* : for brain preprocessing

---

## 3. Data 📀
Rawdata from some participants (incluing A006) can be found in the openneuro dataset [here](https://openneuro.org/datasets/ds005075/).
Preprocessed data from a single participant can be found here (A006):
Denoised data from a single participant can be found here (A006):

Project folder structure:
```
.   
├── dataset_description.json
├── rawdata
    └── sub-A006
│       ├── anat
│       ├── dwi
│       └── func
└── derivatives
    └── Aging_Project
        ├── denoising
        │   └── slice_wise
        └── preprocessing
            ├── sub-A006
            │   ├── anat
            │   ├── dwi
            │   └── func
            └── Aging_Project
                ├── code
                │   ├── connectivity
                │   └── spm
                ├── config
                │   ├── analyses
                │   └── preprocessing
                ├── notebook
                │   ├── main_figures
                │   ├── preprocessing
                │   └── suppl_figures
                └── template
                   ├── MNI
                   └── PAM50
```
---

## 4. Run the Analysis Pipeline ⚙️
<details><summary>Here is a brief description of the files used for data analysis.</summary>

- **`CL_brsc_aging_env.sh`**:: project's environment
- **`code/`**: Functions and code to run the analyses. Do not modify the file.- */code/spm/* : this folder contain the matlab scripts used with SPM
    - **`code/SPM`**: this folder contain the matlab scripts used with SPM
- **`config/`**: Configuration files for paths and parameters.
  - **`config/preprocessing/*.json`**: config files used for preprocessing (i.e used by the notebooks in notebook/preprocessing/)
  - `participants.tsv` contains demographical information and important info for preprocessing (*e.g.,* slice number for vertebrae labeling initiation)
- **`template`**: Used for analyses; do not modify.
- **`notebook`**: The notebook folder contain the different notebooks used to run and vizualized the results. Notebook's number correspond to the related figure number
    - **`notebook/preprocessing/`** : notebooks used for preprocessing
    - **`notebook/main_figures/`** : notebooks used to generate the main figures
    - **`notebook/suppl_figures/`** : notebooks used to generate the supplementary figures
    - **`notebook/preprocessing/`** : notebooks used for preprocessing

</details>

## 4.1 Preprocessing notebooks:
*For details on the preprocessing steps, please refer to the notes within each notebook.*
- Run the codes in the following order:
    - `01a_brsc_preprocess_func.ipynb` > functional and T1w preprocessing
    - `01b_brsc_preprocess_dartel.ipynb` > brain DARTEL template creation

> ⚠️  Some steps need to be manually corrected, this will imply that all subsequent steps need to be re-run.

Then you can run in parallel the following notebooks for each type of contrast:
- `02_brsc_denoising.ipynb` > functional denoising
- `03_sc_preprocess_microstructural.ipynb` > microstructural preprocessing
- `04_sc_preprocess_diffusion.ipynb` > diffusion preprocessing

## 4.2 Main figure notebooks:
*For details on the preprocessing steps, please refer to the notes within each notebook.*
The figure number correspond to the notebook number:
- Figure 1: `Fig01_SpiMorpho.ipynb`
- Figure 2, four notebooks to be run in order:
    - `Fig02a_SpiDyn-BrainDyn_alff.ipynb` > Compute alff
    - `Fig02a_SpiDyn.ipynb` > Compute SpiDyn features and age-related changes. 
    - `Fig02b_SpiFC.ipynb` > Compute functional connectivity analyses and its age-related changes
    - `Fig02c_SpicFC-SpiDyn_coupling.ipynb` > Compute SpinDyn-SpiFC coupling and its age-related changes
- Figure 3:
    -`Fig03_structure-function_coupling.ipynb` > Compute structural-functional coupling and its age-related changes 
- Figure 4, three notebooks to be run in order:
    - `Fig04a_brain-morpho.ipynb` > Compute brain morpho and its age-related changes (Fig 4A-B)
    - `Fig04b_BrainFC-SpiFC.ipynb` > Compute brain FC and its age-related changes (Fig 4C-D)
    - `Fig04c_BrainDyn-SpiDyn.ipynb` > Compute brain dyn and its age-related changes (Fig 4E-F)

## 4.3 Supplementary figure notebooks:    
- Figure S2: `FigS2_QC.ipynb`
- Figure S3: `FigS3_iCAPs.ipynb`
- Figure S5: `S05_morphometry.ipynb` > tract-specific analyses
- Figure S6: `S08_functional_temporal.ipynb` > Distribution of the SpiDyn features