def save_lora_adapter_hook(models, weights, output_dir, accelerator):
    if accelerator.is_main_process:
        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            # Save model params
            trainable_sd = {
                k: v
                for k, v in unwrapped.named_parameters()
                if v.requires_grad
            }

            save_file(trainable_sd, os.path.join(output_dir, 'model.safetensors'))
            if weights:
                weights.pop()