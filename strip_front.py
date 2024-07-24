import torch
from diffusers import StableDiffusionPipeline

# Load the pipeline
model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Print the main components of the pipeline
print("Main components:")
for name, module in pipe.components.items():
    print(f"- {name}")

# Inspect the UNet model
print("\nUNet architecture:")
for name, module in pipe.unet.named_modules():
    if len(list(module.children())) == 0:  # Only print leaf modules
        print(f"- {name}")

# Define a new model class with only the desired layers
class StrippedUNet(torch.nn.Module):
    def __init__(self, original_unet):
        super(StrippedUNet, self).__init__()
        self.conv_in = original_unet.conv_in
        self.time_proj = original_unet.time_proj
        self.time_embedding = original_unet.time_embedding

    def forward(self, x, timesteps=None, context=None):
        # Define the forward pass for the stripped model
        x = self.conv_in(x)
        emb = self.time_proj(timesteps)
        emb = self.time_embedding(emb)
        return x, emb

# Instantiate the stripped model
stripped_unet = StrippedUNet(pipe.unet)

# Save the stripped model
torch.save(stripped_unet.state_dict(), 'stripped_unet.pth')

print("Stripped UNet model saved successfully.")

# Define a new model class with the remaining layers
class RemainingUNet(torch.nn.Module):
    def __init__(self, original_unet):
        super(RemainingUNet, self).__init__()
        self.down_blocks = original_unet.down_blocks
        self.mid_block = original_unet.mid_block
        self.up_blocks = original_unet.up_blocks
        self.conv_norm_out = original_unet.conv_norm_out
        self.conv_out = original_unet.conv_out

    def forward(self, x, timesteps=None, context=None):
        # Forward pass for the remaining layers
        for down_block in self.down_blocks:
            x = down_block(x, emb, context)
        x = self.mid_block(x, emb, context)
        for up_block in self.up_blocks:
            x = up_block(x, emb, context)
        x = self.conv_norm_out(x)
        x = self.conv_out(x)
        return x

# Instantiate the remaining model
remaining_unet = RemainingUNet(pipe.unet)

# Save the remaining model
torch.save(remaining_unet.state_dict(), 'remaining_unet.pth')

print("Remaining UNet model saved successfully.")