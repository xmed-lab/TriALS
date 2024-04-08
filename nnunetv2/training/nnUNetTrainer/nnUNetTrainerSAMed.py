from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from nnunetv2.nets.sam_lora_image_encoder import LoRA_Sam
from monai.networks.nets import SwinUNETR
from nnunetv2.nets.segment_anything.modeling.mask_decoder import MLP, MaskDecoder
from nnunetv2.nets.segment_anything import sam_model_registry
from nnunetv2.training.lr_scheduler.samedlr import CustomWarmupDecayLR
from monai.transforms import (
    Resize,

)
from torch._dynamo import OptimizedModule

from typing import Union


class nnUNetTrainerSAMed(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 2 ** 5) < 1 or ((original_patch_size[i] / 2 ** 5) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 2 ** 5 + 0.5) * 2 ** 5
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.print_to_log_file("Patch size changed from {} to {}".format(original_patch_size, new_patch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size
        self.initial_lr = 1e-3
        self.weight_decay = 0.01
        self.lr_decay=0.9

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            low_res_label_batch = [self.resize(i.to(self.device, non_blocking=True).squeeze()) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            low_res_label_batch = self.resize(target.squeeze())

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            outputs = self.network(data, True, self.patch_size)
        # print(outputs['low_res_logits'].size(), low_res_label_batch.size(),self.label_manager.has_regions)
        # print(torch.unique(low_res_label_batch),)
            l = self.loss(outputs['low_res_logits'], low_res_label_batch.unsqueeze(1))

        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            low_res_label_batch = [self.resize(i.to(self.device, non_blocking=True).squeeze()) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            low_res_label_batch = self.resize(target.squeeze())

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast is a little ****.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        output = self.network(data,True, self.patch_size)
        del data

        l = self.loss(output['low_res_logits'], low_res_label_batch.unsqueeze(1))
        output_masks = output['masks']

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output_masks.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_masks) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output_masks.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_masks.shape, device=output_masks.device,
                                                        dtype=torch.float32)
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

    # def calc_loss(self,outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    #     low_res_logits = outputs['low_res_logits']
    #     loss_ce = ce_loss(low_res_logits, low_res_label_batch.long())
    #     loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    #     loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    #     return loss, loss_ce, loss_dice

    # %%

    def configure_optimizers(self):

        # Custom scheduler setup
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.initial_lr,
                          betas=(0.9, 0.999),
                          weight_decay=0.1)
        scheduler = CustomWarmupDecayLR(optimizer, warmup_period=10, max_iterations=self.num_epochs,
                                        base_lr=self.initial_lr, weight_decay=self.lr_decay)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.get_lora_parameters(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_lora_parameters(new_state_dict)
            else:
                self.network.module.load_lora_parameters(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_lora_parameters(new_state_dict)
            else:
                self.network.load_lora_parameters(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])


class nnUNetTrainerV2_SAMed_h_r_4(nnUNetTrainerSAMed):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.patch_size = 512
        self.resize = Resize(spatial_size=(128, 128), mode='nearest')
        # self.configuration_manager.patch_size=[self.patch_size, self.patch_size]
        self.lr_decay=7
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)

        sam, img_embedding_size = sam_model_registry['vit_h'](image_size=512,
                                                              num_classes=8,  # To load LoRA weights
                                                              checkpoint='checkpoints/sam_vit_h_4b8939.pth',
                                                              pixel_mean=[0, 0, 0],
                                                              pixel_std=[1, 1, 1])
        model = LoRA_Sam(sam, 4)
        # net.load_lora_parameters('checkpoints/epoch_299.pth')
        model.sam.mask_decoder = MaskDecoder(transformer=model.sam.mask_decoder.transformer,
                                             transformer_dim=model.sam.mask_decoder.transformer_dim,
                                             num_multimask_outputs=label_manager.num_segmentation_heads-1 #remove bg
                                             )

        return model

class nnUNetTrainerV2_SAMed_h_r_4_100epochs(nnUNetTrainerV2_SAMed_h_r_4):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_epochs = 100

class nnUNetTrainerV2_SAMed_b_r_4(nnUNetTrainerSAMed):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.patch_size = 256
        self.resize = Resize(spatial_size=(64, 64), mode='nearest')

        # self.configuration_manager.patch_size=[self.patch_size, self.patch_size]
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)

        sam, img_embedding_size = sam_model_registry['vit_b'](image_size=256,
                                                              num_classes=8,  # To load LoRA weights
                                                              checkpoint='checkpoints/sam_vit_b_01ec64.pth',
                                                              pixel_mean=[0, 0, 0],
                                                              pixel_std=[1, 1, 1])
        model = LoRA_Sam(sam, 4)
        # net.load_lora_parameters('checkpoints/epoch_299.pth')
        model.sam.mask_decoder = MaskDecoder(transformer=model.sam.mask_decoder.transformer,
                                             transformer_dim=model.sam.mask_decoder.transformer_dim,
                                             num_multimask_outputs=label_manager.num_segmentation_heads-1 #remove bg
                                             )
        return model

class nnUNetTrainerV2_SAMed_b_r_4_100epochs(nnUNetTrainerV2_SAMed_b_r_4):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
