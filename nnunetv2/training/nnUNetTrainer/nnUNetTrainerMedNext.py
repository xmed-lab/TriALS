from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from nnunetv2.nets.mednextv1.MedNextV1 import MedNeXt
from monai.networks.nets import SwinUNETR

class nnUNetTrainerMedNext(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 2**5) < 1 or ((original_patch_size[i] / 2**5) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 2**5 + 0.5) * 2**5
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.print_to_log_file("Patch size changed from {} to {}".format(original_patch_size, new_patch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size

        self.grad_scaler = None
        self.initial_lr = 1e-3
        self.weight_decay = 0.01

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerV2_MedNeXt_L_kernel5(nnUNetTrainerMedNext):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = create_mednextv1_large(num_input_channels, label_manager.num_segmentation_heads, 5, False)

        return model
class nnUNetTrainerV2_MedNeXt_L_kernel5_100epochs(nnUNetTrainerV2_MedNeXt_L_kernel5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100

class nnUNetTrainerV2_MedNeXt_B_kernel5(nnUNetTrainerMedNext):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = create_mednextv1_base(num_input_channels, label_manager.num_segmentation_heads, 5, False)

        return model

class nnUNetTrainerV2_MedNeXt_B_kernel5_100epochs(nnUNetTrainerV2_MedNeXt_B_kernel5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100



class nnUNetTrainerV2_MedNeXt_M_kernel5(nnUNetTrainerMedNext):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = create_mednextv1_medium(num_input_channels, label_manager.num_segmentation_heads, 5, False)

        return model

def create_mednextv1_small(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=2,
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    )


def create_mednextv1_base(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    )


def create_mednextv1_medium(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style='outside_block'
    )


def create_mednextv1_large(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style='outside_block'
    )


def create_mednext_v1(num_input_channels, num_classes, model_id, kernel_size=3,
                      deep_supervision=False):
    model_dict = {
        'S': create_mednextv1_small,
        'B': create_mednextv1_base,
        'M': create_mednextv1_medium,
        'L': create_mednextv1_large,
    }

    return model_dict[model_id](
        num_input_channels, num_classes, kernel_size, deep_supervision
    )
