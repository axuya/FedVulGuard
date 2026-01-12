# src/federated/client.py
import copy
import torch


class FLClient:
    def __init__(self, name, train_loader, test_loader, num_samples):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_samples = int(num_samples)

    def local_train(
        self,
        global_model,
        train_one_epoch_fn,
        optimizer_fn,
        local_epochs,
        device,
        algo: str = "fedavg",
        mu: float = 0.0,
    ):
        """
        algo: 'fedavg' or 'fedprox'
        mu: FedProx proximal strength (typical: 1e-3 ~ 1e-2)
        """
        model = copy.deepcopy(global_model).to(device)
        opt = optimizer_fn(model)

        # snapshot global params for FedProx proximal term
        global_params = None
        if algo.lower() == "fedprox":
            global_params = [p.detach().clone().to(device) for p in global_model.parameters()]

        last_loss = None
        for _ in range(local_epochs):
            last_loss = train_one_epoch_fn(
                model, self.train_loader, opt,
                device=device,
                algo=algo,
                mu=mu,
                global_params=global_params,
            )
        return model.state_dict(), last_loss
