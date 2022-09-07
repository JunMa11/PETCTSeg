import SimpleITK
import time
import os
import numpy as np
import subprocess
import shutil
import cc3d
import nibabel as nib
#from nnunet.inference.predict import predict_from_folder
from predict import predict_from_folder
from nnunet.inference.ensemble_predictions import merge
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import torch


class Autopet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result'
        self.nii_seg_file = 'TCIA_001.nii.gz'

        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.result_path, self.nii_seg_file), os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """
        #cproc = subprocess.run(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres', shell=True, check=True)
        #os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres')
        print("nnUNet segmentation starting!")
        input_folder = self.nii_path
        output_folder = self.result_path
        part_id = 0  #args.part_id
        num_parts = 1  #args.num_parts
        lowres_segmentations = None  #args.lowres_segmentations
        num_threads_preprocessing = 1  #args.num_threads_preprocessing
        num_threads_nifti_save = 1  # args.num_threads_nifti_save
        disable_tta = True  #args.disable_tta
        step_size = 0.5  #args.step_size
        # interp_order = args.interp_order
        # interp_order_z = args.interp_order_z
        # force_separate_z = args.force_separate_z
        overwrite_existing = False  #args.overwrite_existing
        mode = 'normal'  #args.mode
        all_in_gpu = None  #args.all_in_gpu
        model = '3d_fullres'  # args.model
        trainer_class_name = default_trainer  #args.trainer_class_name
        cascade_trainer_class_name = default_cascade_trainer  # args.cascade_trainer_class_name
        disable_mixed_precision = False  #args.disable_mixed_precision
        plans_identifier = default_plans_identifier
        chk = 'model_best'

        task_name = 'Task001_TCIA'
        start_time = time.time()
        ############################## Model 0 ###############################
        trainer = trainer_class_name
        model_folder_name = join(network_training_output_dir, model, task_name, 'nnUNetTrainerV2__nnUNetPlansv2.1')
        # folds = [0,1,2,3,4]  # args.folds
        save_npz = True  #args.save_npz
        print('Model 1 uses', model_folder_name, 'save_npz=', save_npz)
        # fold 0
        output_folder_f0 = join(output_folder, "model0_npz")
        predict_from_folder(model_folder_name, input_folder, output_folder_f0, [1,2,3], save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk)

        ############################## Model 1 ###############################
        model_folder_name = join(network_training_output_dir, model, task_name, 'nnUNetTrainerV2__nnUNetPlansv2.2')
        save_npz = True 
        print('Model 2 uses', model_folder_name, 'save_npz=', save_npz)
        # fold 0
        output_folder_f1 = join(output_folder, "model1_npz")
        predict_from_folder(model_folder_name, input_folder, output_folder_f1, [1,2,3], save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk)


        ############################## Model 2 ###############################
        model_folder_name = join(network_training_output_dir, model, task_name, 'nnUNetTrainerV2__nnUNetPlansv2.3')
        save_npz = True 
        print('Fold 2 uses', model_folder_name, 'save_npz=', save_npz)
        # fold 0
        output_folder_f2 = join(output_folder, "model2_npz")
        predict_from_folder(model_folder_name, input_folder, output_folder_f2, [1,3], save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk)


        ############################## Model 3 ###############################
        model_folder_name = join(network_training_output_dir, model, task_name, 'nnUNetTrainerV2__nnUNetPlansv2.4')
        save_npz = True 
        print('Fold 3 uses', model_folder_name, 'save_npz=', save_npz)
        # fold 0
        output_folder_f3 = join(output_folder, "model3_npz")
        predict_from_folder(model_folder_name, input_folder, output_folder_f3, [1,2,3], save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk)

        ############################## Model 4 ###############################
        model_folder_name = join(network_training_output_dir, model, task_name, 'nnUNetTrainerV2__nnUNetPlansv2.5')
        save_npz = True 
        print('Fold 4 uses', model_folder_name, 'save_npz=', save_npz)
        # fold 0
        output_folder_f4 = join(output_folder, "model4_npz")
        predict_from_folder(model_folder_name, input_folder, output_folder_f4, [2,3], save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk)

        ############################## Model Ensemble ###############################
        pred_folders = [output_folder_f0, output_folder_f1, output_folder_f2, output_folder_f3, output_folder_f4]
        output_folder_temp = join(output_folder, "temp")
        merge(pred_folders, output_folder_temp, threads=2, override=True, postprocessing_file=None, store_npz=False)
        print('Total running time:', time.time()-start_time)
        print('*'*50)

        ############################## Waiting for exporting results ###############################
        while not os.path.exists(os.path.join(output_folder_temp, self.nii_seg_file)):
            print('.', end='')
            time.sleep(5)
        # post processing
        nii = nib.load(os.path.join(output_folder_temp, self.nii_seg_file))
        data = nii.get_fdata()
        if np.sum(data)>10:
            ct_data = nib.load(os.path.join(self.nii_path, self.nii_seg_file.split('.nii.gz')[0]+'_0000.nii.gz')).get_fdata()
            seg_conn_comp = cc3d.connected_components(data, connectivity=18)
            for con_id in range(1, seg_conn_comp.max()+1):
                comp_mask = np.isin(seg_conn_comp, con_id).astype(np.uint8)
                ct_mask_values = ct_data[comp_mask>0]
                if np.std(ct_mask_values) < 5 or np.mean(ct_mask_values) < -1000:
                    data[comp_mask>0] = 0
            save_nii = nib.Nifti1Image(data, nii.affine, nii.header)
            nib.save(save_nii, os.path.join(output_folder, self.nii_seg_file))
        else:
            print('No lesions found!')
            new_data = np.zeros_like(data)
            save_nii = nib.Nifti1Image(new_data, nii.affine, nii.header)
            nib.save(save_nii, os.path.join(output_folder, self.nii_seg_file))
        print('Prediction finished')

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        self.predict()
        print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()