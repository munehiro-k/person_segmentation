import json
import os
import shutil
import time
from enum import Enum, auto

import numpy as np
import pandas as pd
# import pandas as pd
import torch
import torch.optim
import torch.utils.data as data
import torch_optimizer as optim
from model import SegmentationNet
from utils import load_images

import losses
from person_segmentation.dataset import DATA_TRANSFORMS, AugmentedTrainDataset, TestDatasetMemory

torch.autograd.set_detect_anomaly(True)


class TuningPart(Enum):
    NO_TUNING = auto()
    FINE_TUNING = auto()
    MORE = auto()
    ALL = auto()


class OptimizerType(Enum):
    SCHEDULER = auto()
    ADAPTIVE = auto()


with open("./global_params.json", "r") as f:
    parameters = json.load(f)
local_parameters = {
    "model_type": "_segmentation",
    "batch_size": 25,
    "loss_weight": [1., 1., 4., 100.],
    "optimizer_type": OptimizerType.ADAPTIVE,
    "adaptive_params": {
        "lr": 1e-2
    },
    "scheduler_params": {
        "lr": 5e-2,
        "period": 40,
        "decay_per_period": 0.6
    },
    "tuning_part": TuningPart.ALL,
    "num_epochs": 40
}
parameters.update(local_parameters)


if __name__ == '__main__':
    csv_path = parameters["csv_path"]
    train_background_dir = parameters["train_background_dir"]
    validation_background_dir = parameters["validation_background_dir"]
    model_type = parameters["model_type"]
    load_model_path = parameters["load_model_path"].format(model_type)
    save_model_path = parameters["save_model_path"].format(model_type)
    save_latest_model_path =\
        parameters["save_latest_model_path"].format(model_type)
    save_dir = os.path.dirname(save_model_path)
    backup_model_path = parameters["backup_model_path"].format(model_type)
    for path in (save_model_path, save_latest_model_path, backup_model_path):
        dir_ = os.path.dirname(path)
        os.makedirs(dir_, exist_ok=True)
    size = tuple(parameters["size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(csv_path, header=0, index_col=None)
    df_train = df[df["training"] == 1]
    ret = load_images(df_train, train_background_dir)
    train_images, train_masks, train_backgrounds = ret

    df_test = df[df["training"] == 0]
    ret = load_images(df_test,
                      validation_background_dir)
    validation_images, validation_masks, validation_backgrounds = ret

    net = SegmentationNet()
    if load_model_path:
        if os.path.isfile(load_model_path):
            net.load_state_dict(torch.load(load_model_path,
                                           map_location=device),
                                strict=False)
        else:
            print((f"warning: model weight {load_model_path} does not exist. "
                   "just initialize weights and resume training"))
    net = net.to(device)

    part = parameters["tuning_part"]
    for name, param in net.named_parameters():
        cond = False
        if part == TuningPart.NO_TUNING:
            pass
        elif part == TuningPart.FINE_TUNING:
            if ((name.startswith("decoder.") or
                    name.startswith("segmentation_head.") or
                    name.startswith("input_bn."))):
                cond = True
        elif part == TuningPart.MORE:
            if ((name.startswith("net.encoder.model.conv_stem.") or
                    name.startswith("net.encoder.model.bn1.") or
                    name.startswith("encoder.model.blocks.5.") or
                    name.startswith("decoder.") or
                    name.startswith("segmentation_head.") or
                    name.startswith("input_bn."))):
                cond = True
        else:
            cond = True

        param.requires_grad = cond
        if cond:
            net.params_to_update.append(param)

    batch_size = parameters["batch_size"]
    train_dataset = AugmentedTrainDataset(
        train_images, train_masks, size=size,
        transform=DATA_TRANSFORMS["train"]
    )
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=2,
                                       pin_memory=True,
                                       shuffle=True)

    validation_dataset = TestDatasetMemory(
        validation_images, validation_masks,
        size=size, transform=DATA_TRANSFORMS["val"]
    )
    validation_dataloader = data.DataLoader(validation_dataset,
                                            batch_size=batch_size,
                                            num_workers=2,
                                            pin_memory=True)

    criterions = [
        losses.SigmoidBinaryFocalLoss(
            alpha=None, normalized=True, sum_reduce_dim=(-2, -1),
            thres=0.9, coeff=None
        ),
        losses.CorrelationCoefficientLoss(from_logits=True),
        losses.BDBinaryFocalLoss(
            normalized=True, thres=0.9, coeff=None
        ),
        losses.ModifiedBDLoss(
            from_logits=True, weighted=True, cosine_similarity=False,
            log_scale=True, margin=1.
        )
    ]
    criterion_names = [type(c).__name__ for c in criterions]
    loss_weight = parameters["loss_weight"]
    if parameters["optimizer_type"] == OptimizerType.SCHEDULER:
        lr = parameters["scheduler_params"]["lr"]
        period = parameters["scheduler_params"]["period"]
        decay = np.power(parameters["scheduler_params"]["decay_per_period"],
                         1. / period)
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True)
        lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=period // 2,
            eta_min=(0.01 * lr))
        lr_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=decay)
    else:
        optimizer = optim.Yogi(
            net.parameters(),
            lr=parameters["adaptive_params"]["lr"],
            betas=(0.9, 0.999),
            eps=1e-6,
            initial_accumulator=1e-6,
            weight_decay=0,
        )
    num_epochs = parameters["num_epochs"]
    train_size = len(train_dataset)
    test_size = len(validation_dataset)
    best_metric = np.inf
    lr = []
    training_loss = []
    training_metrics = [[] for _ in range(len(criterions))]
    validation_loss = []
    validation_metrics = [[] for _ in range(len(criterions))]
    start_perf = time.perf_counter()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-------------', flush=True)

        # train
        running_total_loss = 0.0
        running_loss = [0.0 for _ in range(len(criterions))]
        net.train()
        nbatch = 0
        for _X, _y in train_dataloader:
            optimizer.zero_grad()
            _X, _y = _X.to(device), _y.to(device)
            outputs = net(_X)
            loss = torch.tensor([0.], device=device, dtype=torch.float32)
            for i in range(len(criterions)):
                individual_loss = criterions[i](outputs.squeeze(), _y)
                loss += loss_weight[i] * individual_loss
                running_loss[i] += individual_loss.item()
            loss.backward()
            optimizer.step()
            nbatch += 1

            running_total_loss += loss.item()
        running_total_loss = running_total_loss / nbatch
        print((f'training loss: '
               f'{running_total_loss:.5g}'))
        for i in range(len(running_loss)):
            running_loss[i] /= nbatch
            print(f'training {criterion_names[i]}: {running_loss[i]:.5g}')
            training_metrics[i].append(running_loss[i])
        training_loss.append(running_total_loss)
        for param_group in optimizer.param_groups:
            lr.append(param_group['lr'])
            print(f'learning rate: {param_group["lr"]:.5g}')

        # 学習率調整
        if parameters["optimizer_type"] == OptimizerType.SCHEDULER:
            gamma = decay
            lr_scheduler1.step()
            lr_scheduler2.step()
            lr_scheduler1.eta_min *= gamma
            lr_scheduler1.base_lrs = [val * gamma for val
                                      in lr_scheduler1.base_lrs]

        # val
        running_total_loss = 0.0
        running_loss = [0.0 for _ in range(len(criterions))]
        nbatch = 0
        net.eval()
        with torch.no_grad():
            for _X, _y in validation_dataloader:
                _X, _y = _X.to(device), _y.to(device)
                outputs = net(_X)
                loss = torch.tensor([0.], device=device, dtype=torch.float32)
                for i in range(len(criterions)):
                    individual_loss = criterions[i](outputs.squeeze(), _y)
                    loss += loss_weight[i] * individual_loss
                    running_loss[i] += individual_loss.item()
                running_total_loss += loss.item()
                nbatch += 1
        running_total_loss = running_total_loss / nbatch
        print((f'validation loss: '
               f'{running_total_loss:.5g}'))
        for i in range(len(running_loss)):
            running_loss[i] /= nbatch
            print(f'validation {criterion_names[i]}: {running_loss[i]:.5g}')
            validation_metrics[i].append(running_loss[i])
        validation_loss.append(running_total_loss)

        torch.save(net.state_dict(), save_latest_model_path)
        # save best model
        running_metric = running_total_loss
        if running_metric < best_metric:
            best_metric = running_metric
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(net.state_dict(), save_model_path)

    if load_model_path:
        if os.path.isfile(load_model_path):
            shutil.copy2(load_model_path, backup_model_path)
        torch.save(net.state_dict(), load_model_path)

    elapsed_perf = time.perf_counter() - start_perf
    elapsed_min = elapsed_perf / 60
    print((f"elapsed time: {elapsed_min:.1f} [min] "
           f"for {num_epochs} epochs "
           f"({elapsed_perf / num_epochs:.2f} [sec] per epoch)"))
