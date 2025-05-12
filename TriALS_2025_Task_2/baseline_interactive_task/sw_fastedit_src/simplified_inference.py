from monai.networks.nets.dynunet import DynUNet
from monai.inferers import SlidingWindowInferer
import torch
from sw_fastedit.data import get_test_loader, get_pre_transforms_val_as_list, get_click_transforms_json, get_post_transforms
from monai.transforms import Compose
from sw_fastedit.api import init
from sw_fastedit.utils.argparser import parse_args, setup_environment_and_adapt_args
import os
import shutil
import time

        

def simplified_predict(input_folder, output_folder, json_dict, docker=True):
    args = parse_args()

    args.input_dir = input_folder
    args.json_dir = json_dict
    args.output_dir = output_folder

    liver_folder = os.path.join(os.path.dirname(input_folder), 'livers')
    os.makedirs(liver_folder, exist_ok=True)
    if not os.path.exists(os.path.join(liver_folder, os.path.basename(input_folder))):
        os.system(f'TotalSegmentator -i {input_folder} -o {liver_folder} --roi_subset liver --fast --nr_thr_saving 1')
        time.sleep(10)
        os.rename(os.path.join(liver_folder, 'liver.nii.gz'), os.path.join(liver_folder, os.path.basename(input_folder)))

    setup_environment_and_adapt_args(args)

    init(args)

    network = DynUNet(
        spatial_dims=3,
        in_channels=3, # image + fg + bg = 3
        out_channels=2, # len(labels) = fg + bg = 2
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )

    network.load_state_dict(torch.load(args.resume_from)["net"])

    sw_params = {
        "roi_size": (128, 128, 128),
        "mode": "gaussian",
        "cache_roi_weight_map": True,
    }
    eval_inferer = SlidingWindowInferer(sw_batch_size=1, overlap=0.25, **sw_params)

    device = torch.device('cuda:0') 
    pre_transforms_val = Compose(get_pre_transforms_val_as_list(args.labels, device, args))
    post_transform = get_post_transforms(args.labels, save_pred=True, output_dir=args.output_dir, pretransform=pre_transforms_val, output_postfix=str(args.n_clicks), docker=True)

    val_loader = get_test_loader(args, pre_transforms_val)
    network.eval()
    network.to(device)


    for data in val_loader:
        data['image'] = data['image'].to(device)[0]

        click_transforms = get_click_transforms_json(device, args, n_clicks=args.n_clicks)
        data['image'] = click_transforms(data)['image'].unsqueeze(0) # img + guidance signals (fg + bg)
        with torch.no_grad():
            pred = eval_inferer(inputs=data['image'], network=network)
            data['pred'] = pred[0]
            data['pred'] = post_transform(data)['pred']
    if docker:
        print(f'Predictions done for {len(json_dict["lesion"])} clicks!')

    pred_path = os.path.join(args.output_dir, 'predictions', os.path.basename(args.input_dir).replace('0000', f'0000_{args.n_clicks}'))
    
    if docker:
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

        pred_np, _ = SimpleITKIO().read_images([pred_path])

        shutil.rmtree(os.path.dirname(pred_path)) # clean up temp prediction file

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return pred_np
    


if __name__ == "__main__":
    args = parse_args()

    simplified_predict(args.input_dir, args.output_dir, args.json_dir, docker=False)


        

