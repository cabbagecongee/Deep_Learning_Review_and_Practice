#model.py
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''



class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    #instruct TorchScript not to trace this method -> not to apply weight decay to the class token
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}
        
    def forward(self, features, lorentz_vectors, mask): #omit the points feature since its unused
        return self.mod(features, v=lorentz_vectors, mask=mask)
    # features = particle-level input features, lorentz vectors = 4-momentum info, mask=binary mask for padding
        
#returns an initilized model using cfg dict 
def build_model(cfg):
    init_cfg = {k: v for k, v in cfg.items() if k != 'ckpt_path'}
    _logger.info(f"Building ParT with architecture config: {init_cfg}")
    model = ParticleTransformerWrapper(**init_cfg)
    return model

def load_pretrained_model(model, path, device=None):
    map_location = device if device else 'cpu'
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('state_dict', ckpt) #dict that maps the layers to the model params
    model.load_state_dict(state)
    _logger.info(f"Loaded pretrained model from {path}")
    if device is not None:
        model.to(device)
    return model

def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()