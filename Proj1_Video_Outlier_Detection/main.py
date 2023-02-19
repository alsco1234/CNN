"""
논문 재실험 (Kinetics-400)
- 기존의 방법: 비디오의 첫 segment만 input으로 주어 각 모델마다 segment 길이가 달라 정확한 비교가 어려웠음
- 새로운 방법: 비디오에서 추출한 모든 segment를 input으로 주고 다수결로 해당 비디오의 class를 결정

        TODO
- git issue 확인 (미해결)
- 다른 model softmax와 비교
"""

from omegaconf import OmegaConf
from tqdm import tqdm # 진행 상태바를 보여줌 

from utils import build_cfg_path, form_list_from_user_input, sanity_check
import time
import datetime

def main(args_cli):
    # config
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    # OmegaConf.set_readonly(args, True)
    sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    print(OmegaConf.to_yaml(args))
    if args.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {args.output_path}')
    print('Device:', args.device)

    # import are done here to avoid import errors (we have two conda environements)
    if args.feature_type == 's3d':
        from extract_s3d import ExtractS3D as Extractor
    else:
        raise NotImplementedError(f'Extractor {args.feature_type} is not implemented.')

    extractor = Extractor(args)

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(args.video_paths, args.file_with_video_paths, to_shuffle=True)

    print(f'The number of specified videos: {len(video_paths)}')

    for video_path in tqdm(video_paths):
        extractor._extract(video_path)  # note the `_` in the method name

    # yep, it is this simple!


if __name__ == '__main__':
    start = time.time()
    args_cli = OmegaConf.from_cli()
    main(args_cli)
    end = time.time()

    sec = (end - start)
    result = datetime.timedelta(seconds=sec) #시:분:초.마이크로초
    print(result)
