from transformers import ClapModel, ClapProcessor
import torch

name = "clip-nuraghe-1"

# initialise a model with the right config
initialised_model = ClapModel.from_pretrained("openai/clip-vit-base-patch32")
processor = ClapProcessor.from_pretrained("openai/clip-vit-base-patch32")
# load the fine tuned model weights
state_dict = torch.load(f'../logs/ckpts/{name}/last.ckpt', map_location=torch.device('cpu'))['state_dict']

# only get ema_model.model_ema state dict
state_dict_clean = {}
key = 'model_ema.ema_model'
for state in list(state_dict.keys()):
    if state.startswith(key):
        state_dict_clean[state[len(key) + 1:]] = state_dict[state]

initialised_model.load_state_dict(state_dict_clean)

initialised_model.push_to_hub(f'{name}')
processor.push_to_hub(f'{name}')

    