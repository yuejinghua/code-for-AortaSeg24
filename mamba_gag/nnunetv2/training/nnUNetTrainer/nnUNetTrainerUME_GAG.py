from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import torch
from nnunetv2.nets.UMambaEnc_GAG_normal import get_UME_GAG_3d_from_plans
from batchgenerators.utilities.file_and_folder_operations import join,maybe_mkdir_p
from datetime import datetime
from torch.optim import Adam
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import _LRScheduler

class nnUNetTrainerUME_GAG(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_epochs = 500

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            model = get_UME_GAG_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 3D models are supported")

        return model
