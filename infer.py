# infer.py
import torch
import numpy as np
from model import SimpleNet # Import the network from model.py

def load_model(weights_path):
    """Loads the model architecture and injects the saved weights."""
    model = SimpleNet()
    model.load_state_dict(torch.load(weights_path))
    model.eval() # Set the model to evaluation mode (crucial for inference)
    return model

def predict(model, input_np_array):
    """
    Takes a (256, 256, 3) NumPy array, passes it through the model, 
    and returns a (256, 256, 3) NumPy array.
    """
    # 1. Convert NumPy (H, W, C) to PyTorch Tensor (1, C, H, W)
    input_tensor = torch.tensor(input_np_array.transpose(2, 0, 1)).unsqueeze(0)
    
    # 2. Run inference without tracking gradients (saves memory/compute)
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 3. Convert PyTorch Tensor (1, C, H, W) back to NumPy (H, W, C)
    output_np_array = output_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    
    return output_np_array

if __name__ == "__main__":
    
    weights_file = 'simple_net.pth'
    try:
        my_model = load_model(weights_file)
        print(f"Successfully loaded weights from {weights_file}")
    except FileNotFoundError:
        print(f"Error: {weights_file} not found. Please run train.py first.")
        exit()

    test_input = np.random.rand(256, 256, 3).astype(np.float32)
    print(f"Input array shape: {test_input.shape}")

    test_output = predict(my_model, test_input)
    test_output = np.clip(test_output, 0.0, 1.0)    
    
    print(f"Output array shape: {test_output.shape}")
    print(f"Output type: {type(test_output)}")

    from np_img_conversions import np_to_img

    np_to_img(test_output)
