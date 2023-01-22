import torch

class ModuleDataParallel(torch.nn.DataParallel):
    """This class extends nn.DataParallel to access custom attributes of the module being wrapped
    (by default DataParallel does not allow accessing members after wrapping).
    Read more: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def load_model(load_path, model, optim=None, scaler=None):
    print(f"Loading model from [{load_path}]")
    load_dict = torch.load(load_path)
    model.load_state_dict(load_dict['model_state_dict'])

    if optim:
        optim.load_state_dict(load_dict['optimizer_state_dict'])

    if scaler:
        scaler.load_state_dict(load_dict['scaler_state_dict'])

    batch_id = load_dict['batch_id'] if 'batch_id' in load_dict else None
    return batch_id