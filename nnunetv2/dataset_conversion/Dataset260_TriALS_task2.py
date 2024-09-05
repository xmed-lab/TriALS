import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from generate_dataset_json import generate_dataset_json








if __name__ == '__main__':
    brats_data_dir = '/project/medigentxt/marwan/datasets/nnUNet_raw_data_base'
    brats_data_dir_imgs=join(brats_data_dir, 'Task_2', 'imagesTr')
    brats_data_dir_lbls=join(brats_data_dir, 'Task_2', 'labelsTr (Task 2)')

    task_id = 260
    task_name = "TriALS_task2"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(brats_data_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(brats_data_dir_imgs, prefix='TriALS', join=False)

    for c in case_ids:
        print(c)
        shutil.copy(join(brats_data_dir_imgs, c, c + "_nocontrast.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir_imgs, c, c + "_nocontrast.nii.gz"), join(imagestr, c + '_nocontrast_0000.nii.gz'))
        shutil.copy(join(brats_data_dir_imgs, c, c + "_arterial.nii.gz"), join(imagestr, c + '_arterial_0000.nii.gz'))
        shutil.copy(join(brats_data_dir_imgs, c, c + "_venous.nii.gz"), join(imagestr, c + '_venous_0000.nii.gz'))
        shutil.copy(join(brats_data_dir_imgs, c, c + "_delayed.nii.gz"), join(imagestr, c + '_delayed_0000.nii.gz'))

        shutil.copy(join(brats_data_dir_lbls, c, c + "_combined.nii.gz"), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(brats_data_dir_lbls, c, c + "_nocontrast.nii.gz"), join(labelstr, c + '_nocontrast.nii.gz'))
        shutil.copy(join(brats_data_dir_lbls, c, c + "_arterial.nii.gz"), join(labelstr, c + '_arterial.nii.gz'))
        shutil.copy(join(brats_data_dir_lbls, c, c + "_venous.nii.gz"), join(labelstr, c + '_venous.nii.gz'))
        shutil.copy(join(brats_data_dir_lbls, c, c + "_delayed.nii.gz"), join(labelstr, c + '_delayed.nii.gz'))


    generate_dataset_json(out_base,
                          channel_names={0: 'CT'},
                          labels={
                              'background': 0,
                              'liver': 1,
                              'lesion': 2,
                          },
                          num_training_cases=len(case_ids)*5,
                          file_ending='.nii.gz',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0'
                          )
