from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
join = os.path.join

import nibabel as nb
from skimage import measure, morphology

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

min_organ_size = {'1':7257, '2':16161, '3':5042, '4':621, '5':209,
                  '6':326703, '7':15000, '8':10719, '9':13658, '10':7606,
                  '11':287, '12':152, '13':8496, '14':7332, '15':246}

if __name__ == '__main__':
    """
    This inference script is intended to be used within a Docker container as part of the AMOS Test set submission. It
    expects to find input files (.nii.gz) in /input and will write the segmentation output to /output. Note that this
    guide draws heavily on Kits21's submission guidance, and we are grateful to the project's developers.
    
    IMPORTANT: This script performs inference using one nnU-net configuration (3d_lowres, 3d_fullres, 2d OR 
    3d_cascade_fullres). Within the /parameter folder, nnU-Net expects to find fold_X subfolders where X is the fold ID 
    (typically [0-4]). These folds CANNOT originate from different configurations. There also needs to be the plans.pkl 
    file that you find along with these fold_X folders in the corresponding nnunet training output directory.
    
    /parameters/
    ├── fold_0
    │    ├── model_final_checkpoint.model
    │    └── model_final_checkpoint.model.pkl
    ├── fold_1
    ├── ...
    ├── plans.pkl
    
    Note: nnU-Net will read the correct nnU-Net trainer class from the plans.pkl file. Thus there is no need to 
    specify it here. For the ensembling of different nnU-Net configurations (3d_lowres, 3d_fullres, ...), please refer
    to https://github.com/neheller/kits21/blob/master/examples/submission/nnUNet_submission/run_inference_ensembling.py
    
    IMPORTANT: this script performs inference using nn-UNet project, if users use other codebase, please follow
    dockerfile to install/add required packages, codes. And modify the inference code below.
    """
    #
input_folder = './input'
output_folder = './output_temp'
final_folder = './output'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(final_folder, exist_ok=True)
model_folder_name = './parameters/Task301_AMOSCT/'

from nnunet.inference.predict import predict_cases
from batchgenerators.utilities.file_and_folder_operations import subfiles, join

input_files = subfiles(input_folder, suffix='.nii.gz', join=False)
output_files = [join(output_folder, i) for i in input_files]
input_files = [join(input_folder, i) for i in input_files]

# in the parameters folder are five models (fold_X) traines as a cross-validation. We use them as an ensemble for
# prediction
folds = (0, 1, 2, 3, 4)

# setting this to True will make nnU-Net use test time augmentation in the form of mirroring along all axes. This
# will increase inference time a lot at small gain, so you can turn that off
do_tta = True

print(model_folder_name, output_files)
predict_cases(model_folder_name, [[i] for i in input_files], output_files, folds, save_npz=False,
              num_threads_preprocessing=1, num_threads_nifti_save=1, segs_from_prev_stage=None, do_tta=do_tta,
              mixed_precision=True, overwrite_existing=True, all_in_gpu=False, step_size=0.5, checkpoint_name='model_best')
if os.path.isfile(join(output_folder, 'plans.pkl')):
    os.remove(join(output_folder, 'plans.pkl'))

names = sorted(os.listdir(output_folder))
for name in names:
    nii = nb.load(join(output_folder, name))
    data = nii.get_fdata()
    data = np.uint8(data)
    label_values = np.unique(data)
    for i in label_values[1:]:
        if np.sum(data==i)*np.prod(nii.header.get_zooms()) < min_organ_size[str(i)]:
            data[data==i] = 0
            print(name, 'remove segmentation error label', i)
    #%% clean kidneys
    label_values = np.unique(data)
    if (2 in label_values) and (3 in label_values):
        kidneys = np.uint8(np.logical_or(data==2, data==3))
        data[kidneys>0] = 0
        
        kidneys_1 = getLargestCC(kidneys>0)
        kidneys_1_prop = measure.regionprops(kidneys_1.astype(np.uint8))
        kidneys[kidneys_1] = 0
        kidneys_2 = getLargestCC(kidneys>0)
        kidneys_2_prop = measure.regionprops(kidneys_2.astype(np.uint8))
        if kidneys_1_prop[0]['centroid'][0] > kidneys_2_prop[0]['centroid'][0]:
            data[kidneys_1] = 3 # img-right kidney
            data[kidneys_2] = 2 # img-left kidney
        else:
            data[kidneys_1] = 2
            data[kidneys_2] = 3   
    
    #%% clean other organs
    new_label = np.zeros_like(data, dtype=np.uint8)
    for i in label_values[1:]:
        organ_i = getLargestCC(data==i)
        new_label[organ_i] = i
    save_nii = nb.Nifti1Image(new_label.astype(np.uint8), nii.affine, nii.header)
    nb.save(save_nii, join(final_folder, name))

