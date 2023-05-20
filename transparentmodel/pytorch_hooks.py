from transparentmodel.analysis import gradients_size, activations_size
def backward_hook_gradients(model, grad_input, grad_output):
    print(f"Gradient_size:{gradients_size(grad_input)}")
    # Print or do anything else with the gradient size information

def forward_hook_activations(model, input, output):
    print(f"Activations Size:{activations_size(output['logits'])}")
    # Print or do anything else with the gradient size information