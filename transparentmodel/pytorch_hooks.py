from transparentmodel.analysis import gradients_size, activations_size


def backward_hook_gradients_general(model, grad_input, grad_output):
    print(f"Gradient_size:{gradients_size(grad_input)}")


def forward_hook_activations_hf(model, input, output):
    print(f"Activations Size:{activations_size(output['logits'])}")


def forward_hook_activations_torch(model, input, output):
    print(f"Activations Size:{activations_size(output)}")
