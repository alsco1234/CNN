from typing import Dict

import numpy as np
import torch
import torchvision
from base_extractor import BaseExtractor
from s3d import S3D
from transforms import CenterCrop, Resize, ToFloatTensorInZeroOne
from torchvision.io.video import read_video
#from io import reencode_video_with_diff_fps
from utils import form_slices, show_predictions_on_dataset


class ExtractS3D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        s3d_model_cfgs = {
            'repo': None,
            'stack_size': 64, 'step_size': 64, 'num_classes': 400, 'dataset': 'kinetics'
           
        }
        self.model_def = s3d_model_cfgs
        self.stack_size = 64 if args.stack_size is None else args.stack_size
        self.step_size = 64 if args.step_size is None else args.step_size
        self.extraction_fps = 25 if args.extraction_fps is None else args.extraction_fps
        # normalization is not used as per: https://github.com/kylemin/S3D/issues/4
        self.transforms = torchvision.transforms.Compose([
            ToFloatTensorInZeroOne(),
            Resize(224),
            CenterCrop((224, 224))
        ])
        self.show_pred = args.show_pred
        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
        # take the video, change fps and save to the tmp folder
        #if self.extraction_fps is not None:
        #    video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        rgb, audio, info = read_video(video_path, pts_unit='sec')
        # prepare data (first -- transform, then -- unsqueeze)
        
        rgb = self.transforms(rgb)  # could run out of memory here
        rgb = rgb.unsqueeze(0)
        # slice the stack of frames
        slices = form_slices(rgb.size(2), self.stack_size, self.step_size)

        vid_feats = []

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            rgb_stack = rgb[:, :, start_idx:end_idx, :, :].to(self.device)
            #print(rgb_stack.size()) #torch.Size([1, 3, 64, 224, 224])
            #output = self.name2module['model'](rgb_stack)
            #vid_feats.extend(output.tolist())
            output = rgb_stack

            # video_path 넣는 maybe_show_pred
            self.maybe_show_pred(output, start_idx, end_idx, video_path)

        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        s3d_kinetics400_weights_torch_path = '/Users/kimminchae/Desktop/CNN/S3D_kinetics400_torchified.pt'
        model = S3D(num_class=400, ckpt_path=s3d_kinetics400_weights_torch_path)
        model = model.to(self.device)
        model.eval()

        return {
            'model': model,
        }

    def maybe_show_pred(self, rgb_stack: torch.Tensor, start_idx: int, end_idx: int, video_path): #### video_path 추가함
        if self.show_pred:
            logits = self.name2module['model'](rgb_stack)
            print(f'At frames ({start_idx}, {end_idx})')
            show_predictions_on_dataset(logits, 'kinetics', start_idx/64, video_path) #### start_idx/32 추가함
