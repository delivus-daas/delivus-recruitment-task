import json
import pathlib

import pandas as pd
import torch
from augmentations import embed_data_mask
from sklearn.metrics import mean_absolute_error, mean_squared_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_sq_error(model, dloader, device, vision_dset, opt_use_sep, opt_use_dist):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, vision_dset
            )

            B, C, D = x_cont_enc.shape
            _, F, _ = x_categ_enc.shape

            temp_dim = int(C / 2)
            if opt_use_sep:
                if not opt_use_dist:
                    x_cont_enc = torch.cat(
                        [x_cont_enc[:, :temp_dim, :], x_cont_enc[:, temp_dim:, :]],
                        dim=0,
                    )
                elif opt_use_dist:
                    temp_tensor_1 = torch.cat(
                        [x_cont_enc[:, :temp_dim, :], x_cont_enc[:, temp_dim:-1, :]],
                        dim=0,
                    )
                    temp_tensor_2 = torch.cat(
                        [x_cont_enc[:, -1:, :], x_cont_enc[:, -1:, :]], dim=0
                    )
                    x_cont_enc = torch.cat([temp_tensor_1, temp_tensor_2], dim=1)

            # separate src / dst feature
            if opt_use_sep:
                if F != 1:
                    temp_1 = torch.cat(
                        [
                            x_categ_enc[:, :2, :],
                            x_categ_enc[:, 2 : (2 + int((F - 2) / 2)), :],
                        ],
                        dim=1,
                    )
                    temp_2 = torch.cat(
                        [
                            x_categ_enc[:, :2, :],
                            x_categ_enc[:, (2 + int((F - 2) / 2)) :, :],
                        ],
                        dim=1,
                    )
                    x_categ_enc = torch.cat([temp_1, temp_2], dim=0)
                else:
                    x_categ_enc = torch.cat([x_categ_enc, x_categ_enc], dim=0)

            reps = model.transformer(x_categ_enc, x_cont_enc)
            if opt_use_sep:
                y_reps = torch.cat([reps[:B, :1, :], reps[B:, :1, :]], dim=2)
            elif not opt_use_sep:
                y_reps = reps[:, 0, :]

            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs], dim=0)
        # import ipdb; ipdb.set_trace()
        rmse = mean_squared_error(
            y_test.squeeze().cpu(), y_pred.squeeze().cpu(), squared=False
        )
        return rmse


def mean_ab_error(model, dloader, device, vision_dset, opt_use_sep, opt_use_dist):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, vision_dset
            )

            B, C, D = x_cont_enc.shape
            _, F, _ = x_categ_enc.shape

            temp_dim = int(C / 2)
            if opt_use_sep:
                if not opt_use_dist:
                    x_cont_enc = torch.cat(
                        [x_cont_enc[:, :temp_dim, :], x_cont_enc[:, temp_dim:, :]],
                        dim=0,
                    )
                elif opt_use_dist:
                    temp_tensor_1 = torch.cat(
                        [x_cont_enc[:, :temp_dim, :], x_cont_enc[:, temp_dim:-1, :]],
                        dim=0,
                    )
                    temp_tensor_2 = torch.cat(
                        [x_cont_enc[:, -1:, :], x_cont_enc[:, -1:, :]], dim=0
                    )
                    x_cont_enc = torch.cat([temp_tensor_1, temp_tensor_2], dim=1)

            # separate src / dst feature
            if opt_use_sep:
                if F != 1:
                    temp_1 = torch.cat(
                        [
                            x_categ_enc[:, :2, :],
                            x_categ_enc[:, 2 : (2 + int((F - 2) / 2)), :],
                        ],
                        dim=1,
                    )
                    temp_2 = torch.cat(
                        [
                            x_categ_enc[:, :2, :],
                            x_categ_enc[:, (2 + int((F - 2) / 2)) :, :],
                        ],
                        dim=1,
                    )
                    x_categ_enc = torch.cat([temp_1, temp_2], dim=0)
                else:
                    x_categ_enc = torch.cat([x_categ_enc, x_categ_enc], dim=0)

            reps = model.transformer(x_categ_enc, x_cont_enc)
            if opt_use_sep:
                y_reps = torch.cat([reps[:B, :1, :], reps[B:, :1, :]], dim=2)
            elif not opt_use_sep:
                y_reps = reps[:, 0, :]

            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs], dim=0)
        # import ipdb; ipdb.set_trace()
        mae = mean_absolute_error(y_test.squeeze().cpu(), y_pred.squeeze().cpu())
        return mae


def save_report(directory, report):
    print("Saving Evaluation Report")
    # pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{directory}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report))


def save_baseline(directory, predictions, labels):
    print("Saving Evaluation Quality Baseline")
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    baseline_path = f"{directory}/baseline.csv"
    baseline_dict = {"prediction": predictions, "label": labels}
    pd.DataFrame(baseline_dict).to_csv(baseline_path, header=True, index=False)
