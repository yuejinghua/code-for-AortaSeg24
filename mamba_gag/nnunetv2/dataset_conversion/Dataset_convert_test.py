from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
# from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
# import tifffile
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk

nnUNet_raw = '/hdd2/yjh/nnUNetFranme_V2/nnUNet_raw/'
if __name__ == '__main__':

    dataset_name = 'Dataset264_AortaSeg24_fine'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    for i in os.listdir(imagestr):
        if '_0000' in i:
            ct_name = os.path.join(imagestr,i)
            ct_gt_name = os.path.join(imagestr, i.replace('_0000','_0001'))
            ct_img = sitk.ReadImage(ct_name)
            ct_gt_img = sitk.ReadImage(ct_gt_name)
            origin1 = ct_img.GetOrigin()
            ct_gt_img.SetOrigin(origin1)
            sitk.WriteImage(ct_gt_img, ct_gt_name)
            # origin2 = ct_gt_img.GetOrigin()


    # source_imagesTr = '/hdd2/yjh/nnUNetFranme_V2/nnUNet_raw/Dataset262_AortaSeg24/imagesTr/'
    # source_labelsTr = '/hdd2/yjh/nnUNetFranme_V2/nnUNet_raw/Dataset262_AortaSeg24/labelsTr/'
    # source_labelsTr_4 = '/hdd2/yjh/nnUNetFranme_V2/nnUNet_raw/Dataset263_AortaSeg24_coarse/labelsTr/'

    # shutil.copytree(source_imagesTr,imagestr)
    # shutil.copytree(source_labelsTr, labelstr)
    # for i in os.listdir(source_labelsTr_4):
    #     shutil.copy(os.path.join(source_labelsTr_4,i), os.path.join(imagestr, i.replace('.mha','_0001.mha')))


    # # now we generate the dataset json
    # generate_dataset_json(
    #     join(nnUNet_raw, dataset_name),
    #     {0: 'CT', 1: 'GT'},
    #     {"background": 0,
    #     "zone0": 1,
    #     "innominate": 2,
    #     "zone1": 3,
    #     "left_common_carotid": 4,
    #     "zone2": 5,
    #     "left_subclavian_artery": 6,
    #     "zone3": 7,
    #     "zone4": 8,
    #     "zone5": 9,
    #     "zone6": 10,
    #     "celiac_artery": 11,
    #     "zone7": 12,
    #     "sma": 13,
    #     "zone8": 14,
    #     "right_renal_artery": 15,
    #     "left_renal_artery": 16,
    #     "zone9": 17,
    #     "zone10_r": 18,
    #     "zone10_l": 19,
    #     "right_internal_iliac_artery": 20,
    #     "left_internal_iliac_artery": 21,
    #     "zone11_r": 22,
    #     "zone11_l": 23},
    #     50,
    #     '.mha'
    # )

