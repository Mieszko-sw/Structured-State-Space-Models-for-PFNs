from tabpfn.scripts.model_builder import get_model, save_model

def epoch_callback(model, epoch, config, model_name):

    config["stop_epoch"] = epoch

    print("Saving latest model via epoch callback ...")

    save_model(
        model=model,
        path="./",
        filename=f'tabpfn/models_diff/callback_{model_name}_latest.cpkt',
        config_sample=config
    )

    if epoch % 20 != 0:
        return

    print("Saving numbered model checkpoint via epoch callback ...")

    save_model(
        model=model,
        path="./",
        filename=f'tabpfn/models_diff/callback_{model_name}_epoch_{epoch}.cpkt',
        config_sample=config
    )
