import torch


def build_classifier(model, config, num_classes):
    model.out_head = torch.nn.Linear(
        in_features=config["emb_dim"],
        out_features=num_classes
    )
    return model
