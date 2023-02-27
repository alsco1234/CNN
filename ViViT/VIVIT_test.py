"""
!git clone https://github.com/mx-mark/VideoTransformer-pytorch.git
%cd VideoTransformer-pytorch
"""
import torch
import torch.nn as nn
import numpy as np
import einops
from einops import rearrange, reduce, repeat
from IPython.display import display
import data_transform as T
from dataset import DecordInit, load_annotation_data
from transformer import PatchEmbed, TransformerContainer, ClassificationHead

class ViViT(nn.Module):
    supported_attention_types = [
        'fact_encoder', 'joint_space_time', 'divided_space_time'
    ]

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_p=0.,
                 tube_size=2,
                 conv_type='Conv3d',
                 attention_type='fact_encoder',
                 norm_layer=nn.LayerNorm,
                 return_cls_token=True,
                 **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')

        num_frames = num_frames//tube_size
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.conv_type = conv_type
        self.tube_size = tube_size
        self.num_time_transformer_layers = 4
        self.return_cls_token = return_cls_token

        #tokenize & position embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            tube_size=tube_size,
            conv_type=conv_type)
        num_patches = self.patch_embed.num_patches

        # Divided Space Time Transformer Encoder - Model 2
        transformer_layers = nn.ModuleList([])

        spatial_transformer = TransformerContainer(
            num_transformer_layers=num_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims*4,
            operator_order=['self_attn','ffn'])

        temporal_transformer = TransformerContainer(
            num_transformer_layers=self.num_time_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims*4,
            operator_order=['self_attn','ffn'])

        transformer_layers.append(spatial_transformer)
        transformer_layers.append(temporal_transformer)

        self.transformer_layers = transformer_layers
        self.norm = norm_layer(embed_dims, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
        # whether to add one cls_token in temporal pos_enb
        num_frames = num_frames + 1
        num_patches = num_patches + 1

        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
        self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        self.drop_after_time = nn.Dropout(p=dropout_p)
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        #Tokenize
        b, t, c, h, w = x.shape
        x = self.patch_embed(x)

        # Add Position Embedding
        cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.drop_after_pos(x)

        # fact encoder - CRNN style
        spatial_transformer, temporal_transformer, = *self.transformer_layers,
        x = spatial_transformer(x)

        # Add Time Embedding
        cls_tokens = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
        x = reduce(x, 'b t p d -> b t d', 'mean')
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.time_embed
        x = self.drop_after_time(x)

        x = temporal_transformer(x)

        x = self.norm(x)
        # Return Class Token
        if self.return_cls_token:
            return x[:, 0]
        else:
            return x[:, 1:].mean(1)

def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:]
			state_dict[new_key] = state_dict.pop(old_key)
		else:
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)
                        
def init_from_pretrain_(module, pretrained, init_module):
    if torch.cuda.is_available():
        state_dict = torch.load(pretrained)
    else:
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    if init_module == 'transformer':
        replace_state_dict(state_dict)
    elif init_module == 'cls_head':
        replace_state_dict(state_dict)
    else:
        raise TypeError(f'pretrained weights do not include the {init_module} module')
    msg = module.load_state_dict(state_dict, strict=False)
    return msg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
num_frames = 8
frame_interval = 32
num_class = 400
arch = 'vivit' # turn to vivit for initializing vivit model

pretrain_pth = '/content/drive/My Drive/vivit_model.pth'
num_frames = num_frames * 2
frame_interval = frame_interval // 2
model = ViViT(num_frames=num_frames,
            img_size=224,
            patch_size=16,
            embed_dims=768,
            in_channels=3,
            attention_type='fact_encoder',
            return_cls_token=True)

cls_head = ClassificationHead(num_classes=num_class, in_channels=768)
msg_trans = init_from_pretrain_(model, pretrain_pth, init_module='transformer')
msg_cls = init_from_pretrain_(cls_head, pretrain_pth, init_module='cls_head')
model.eval()
cls_head.eval()
model = model.to(device)
cls_head = cls_head.to(device)
print(f'load model finished, the missing key of transformer is:{msg_trans[0]}, cls is:{msg_cls[0]}')

"""
TEST
"""
from IPython.display import display, HTML

video_path = './demo/YABnJL_bDzw.mp4'
html_str = '''
<video controls width=\"480\" height=\"480\" src=\"{}\">animation</video>
'''.format(video_path)
display(HTML(html_str))

# Prepare data preprocess
mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
data_transform = T.Compose([
        T.Resize(scale_range=(-1, 256)),
        T.ThreeCrop(size=224),
        T.ToTensor(),
        T.Normalize(mean, std)
        ])
temporal_sample = T.TemporalRandomCrop(num_frames*frame_interval)

# Sampling video frames
video_decoder = DecordInit()
v_reader = video_decoder(video_path)
total_frames = len(v_reader)
start_frame_ind, end_frame_ind = temporal_sample(total_frames)
if end_frame_ind-start_frame_ind < num_frames:
    raise ValueError(f'the total frames of the video {video_path} is less than {num_frames}')
frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, num_frames, dtype=int)
video = v_reader.get_batch(frame_indice).asnumpy()
del v_reader

video = torch.from_numpy(video).permute(0,3,1,2) # Video transform: T C H W
data_transform.randomize_parameters()
video = data_transform(video)
video = video.to(device)

"""
VIDEO CLASSIFICATION
"""
# Predict class label
with torch.no_grad():
    logits = model(video)
    output = cls_head(logits)
    output = output.view(3, 400).mean(0)
    cls_pred = output.argmax().item()

class_map = './k400_classmap.json'
class_map = load_annotation_data(class_map)
for key, value in class_map.items():
    if int(value) == int(cls_pred):
        print(f'the shape of ouptut: {output.shape}, and the prediction is: \n{key} == {output.max()}')
        break