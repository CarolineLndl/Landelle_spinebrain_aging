##!/bin/bash

toolbox_home=/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox
anaconda_dir=/export02/data/landelle/anaconda/

# matlab
LD_PREFIX="/export01/local/matlab23b/sys/os/glnxa64:/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/libraries"

export  LD_LIBRARY_PATH=/export01/local/matlab23b/bin/glnxa64/
export TMPDIR=/export02/data/tmp/

# SPINALCORDTOOLBOX (installed on 2020-11-09 09:26:44)
# SPINALCORDTOOLBOX (installed on 2020-11-09 09:26:44)
export PATH="/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.6.0/bin:$PATH"
export SCT_DIR=/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.6.0
export MPLBACKEND=Agg


# python setup
#export PYTHONPATH="${toolbox_home}/sct_4.2.2/spinalcordtoolbox:${toolbox_home}/sct_4.2.2/scripts"
source /export02/data/landelle/anaconda/etc/profile.d/conda.sh
conda activate CL_brsc_aging_env_py10
echo "++ Python version adjusted : `python --version`"

# ANTS setup
export PATH=${PATH}:/usr/lib/ants


# FSL Setup
FSLDIR=/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/fsl
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh


# AFNI configuration
PREFIX=$toolbox_home/afni
if [ -e $PREFIX ] ; then
   if [ -z "$PATH" ] ; then
      export PATH="$PREFIX"
   else
      export PATH="$PREFIX:$PATH"
   fi
   echo "++ AFNI added to PATH: $PREFIX"
fi

#
# R
export PATH="/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/R-4.2.1/lib:$PATH"

