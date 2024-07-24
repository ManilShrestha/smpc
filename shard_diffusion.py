import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch import nn

class CustomUNetWrapper(torch.nn.Module):
    def __init__(self, stripped_unet, remaining_unet):
        super(CustomUNetWrapper, self).__init__()
        self.stripped_unet = stripped_unet
        self.remaining_unet = remaining_unet

    def forward(self, input_tensor, timesteps, encoder_hidden_states=None, **kwargs):
        # First, process input through the stripped UNet
        x, emb = self.stripped_unet(input_tensor, timesteps)

        # Then, pass the output and embeddings to the remaining UNet
        return self.remaining_unet(x, emb, encoder_hidden_states=encoder_hidden_states, **kwargs)


# Load the original pipeline
model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

# Load your custom UNet parts (state dictionaries)
stripped_unet_state_dict = torch.load('stripped_unet.pth')
remaining_unet_state_dict = torch.load('remaining_unet.pth')

# Initialize the custom UNet wrapper with your models
custom_unet = CustomUNetWrapper(stripped_unet_state_dict, remaining_unet_state_dict)

# Instantiate the Stable Diffusion pipeline with the custom UNet
custom_pipeline = StableDiffusionPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=custom_unet,  # Your custom UNet
    scheduler=pipe.scheduler,
    safety_checker=pipe.safety_checker,
    feature_extractor=pipe.feature_extractor
)

# Make sure the pipeline and all components are moved to the correct device
#custom_pipeline = custom_pipeline.to("cuda")


prompt = "portrait photo of an old warrior chief"
generator = torch.Generator(device='cpu').manual_seed(0)

# Generate images using the custom pipeline
output = custom_pipeline(prompt=prompt, generator=generator, num_inference_steps=50)

# Display the generated image
image = output.images[0]
image.show()
