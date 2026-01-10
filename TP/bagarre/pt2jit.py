import torch
import os

# Load the model

for file_name in os.listdir('.'):
    if file_name.endswith('.pt') and 'jit' not in file_name:
        print(f"Processing {file_name}...")
        try:
            model = torch.load(file_name, map_location=torch.device('cpu'), weights_only=False)
            model.eval()  # Set to evaluation mode

            # Convert to TorchScript using tracing
            example_input = torch.randn(1, 1, 8, 8)  # Example input matching model input size
            traced_model = torch.jit.trace(model, example_input)

            # Save the TorchScript model
            output_file = file_name.replace('.pt', '_jit.pt')
            traced_model.save(output_file)

            print(f"Model successfully converted and saved as {output_file}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
