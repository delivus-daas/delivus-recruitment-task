import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def data_split(X, y, indices):
    x_d = {"data": X.values[indices]}
    y_d = {"data": y[indices].reshape(-1, 1)}
    return x_d, y_d


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task="clf", continuous_mean_std=None, uuid=None):
        cat_cols = list(cat_cols)
        X = X["data"].copy()

        # X_mask =  X['mask'].copy()
        X_mask = np.ones_like(X)

        self.uuid_idxs = uuid

        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = (
            X_mask[:, cat_cols].copy().astype(np.int64)
        )  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        if task == "clf":
            self.y = Y["data"]  # .astype(np.float32)
        else:
            self.y = Y["data"].astype(np.float32)

        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return (
            np.concatenate((self.cls[idx], self.X1[idx])),
            self.X2[idx],
            self.y[idx],
            np.concatenate((self.cls_mask[idx], self.X1_mask[idx])),
            self.X2_mask[idx],
            self.uuid_idxs.iloc[idx],
        )


def data_prep_custom(
    df, target_col, seed, task, feat_name_dict, datasplit=[0.65, 0.15, 0.2]
):
    np.random.seed(seed)

    ## Splitting data into features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    uuid_all = X["shippingitem_uuid"]
    X = X.drop(["shippingitem_uuid"], axis=1)

    ## Identifying categorical columns
    # categorical_columns = X.select_dtypes(include=["uint8"]).columns.tolist()
    # cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    categorical_columns = (
        feat_name_dict["common_feat_cat"]
        + feat_name_dict["trg_feat_cat"]
        + feat_name_dict["src_feat_cat"]
    )
    cont_columns = (
        feat_name_dict["common_feat_con"]
        + feat_name_dict["trg_feat_con"]
        + feat_name_dict["src_feat_con"]
    )
    cat_idxs = [X.columns.get_loc(c) for c in categorical_columns if c in X]
    # con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    con_idxs = [X.columns.get_loc(c) for c in cont_columns if c in X]

    X["Set"] = np.random.choice(
        ["train", "valid", "test"], p=datasplit, size=(X.shape[0],)
    )

    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    X = X.drop(columns=["Set"])

    ## Assuming NaNs are already handled, so no need for nan_mask in this custom data.

    cat_dims = [len(X[col].unique()) for col in categorical_columns]
    for col in cont_columns:
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != "regression":
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)

    X_train, y_train = data_split(X, y, train_indices)
    X_valid, y_valid = data_split(X, y, valid_indices)
    X_test, y_test = data_split(X, y, test_indices)

    train_uuid = uuid_all[train_indices]
    valid_uuid = uuid_all[valid_indices]
    test_uuid = uuid_all[test_indices]

    train_mean, train_std = np.array(
        X_train["data"][:, con_idxs], dtype=np.float32
    ).mean(0), np.array(X_train["data"][:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    return (
        cat_dims,
        cat_idxs,
        con_idxs,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        train_mean,
        train_std,
        train_uuid,
        valid_uuid,
        test_uuid,
    )


def load_df(use_cat, opt_use_dist, data):
    # data["sector"] = data["sector"].astype("category").cat.codes
    # if opt_end_date.end_date is not None:
    #     data = data[data["delivery_date"] < opt_end_date]
    #### saving unique feature list
    frequency = data["sector"].value_counts()
    low_frequency_areas = frequency[
        frequency < 10
    ].index.tolist()  # 빈도수 10 미만인 지역 리스트
    data["sector"] = data["sector"].apply(
        lambda x: "기타지역" if x in low_frequency_areas else x
    )

    unique_sector_names = data["sector"].unique().tolist()

    unique_values_dict = {}
    unique_values_dict["sector"] = unique_sector_names

    idx_feat = ["shippingitem_uuid"]
    label_feat = ["deltime"]

    common_feat_con = []
    common_feat_cat = ["sector"]

    trg_feat_con = ["lat", "lng"]
    trg_feat_cat = [
        "postal_digit1",
        "postal_digit2",
        "postal_digit3",
        "postal_digit4",
        "postal_digit5",
    ]

    src_feat_con = ["source_lat", "source_lng"]
    src_feat_cat = [
        "src_postal_digit1",
        "src_postal_digit2",
        "src_postal_digit3",
        "src_postal_digit4",
        "src_postal_digit5",
    ]

    for feature in trg_feat_cat:
        unique_values_dict[feature] = data[feature].unique().tolist()
    with open("json/unique_feature_values.json", "w") as file:
        json.dump(unique_values_dict, file)

    #### TODO: need discussion
    category_to_code = dict(enumerate(data["sector"].astype("category").cat.categories))
    code_to_category = {v: k for k, v in category_to_code.items()}

    data["sector"] = data["sector"].astype("category").cat.codes

    with open("json/code_to_category.json", "w") as file:
        json.dump(code_to_category, file)
    ####

    if not opt_use_dist:
        feature = (
            idx_feat
            + common_feat_con
            + common_feat_cat
            + trg_feat_con
            + src_feat_con
            + trg_feat_cat
            + src_feat_cat
            + label_feat
        )
        df = data[feature]
        cols_to_convert = common_feat_cat + trg_feat_cat + src_feat_cat
        if use_cat:
            df[cols_to_convert] = df[cols_to_convert].astype("uint8")
        else:
            df[cols_to_convert] = df[cols_to_convert].astype("float32")

    elif opt_use_dist:
        feature = (
            idx_feat
            + common_feat_con
            + common_feat_cat
            + trg_feat_con
            + src_feat_con
            + trg_feat_cat
            + src_feat_cat
            + label_feat
        )

        df = data[feature]
        cols_to_convert = common_feat_cat + trg_feat_cat + src_feat_cat

        if use_cat:
            df[cols_to_convert] = df[cols_to_convert].astype("uint8")
        else:
            df[cols_to_convert] = df[cols_to_convert].astype("float32")
    else:
        raise ValueError

    feat_name_dict = {
        "idx_feat": idx_feat,
        "label_feat": label_feat,
        "common_feat_con": common_feat_con,
        "common_feat_cat": common_feat_cat,
        "trg_feat_con": trg_feat_con,
        "trg_feat_cat": trg_feat_cat,
        "src_feat_con": src_feat_con,
        "src_feat_cat": src_feat_cat,
    }
    return df, feat_name_dict


def load_df_inference(use_cat, opt_use_dist, data):

    #### unknown feature processing
    with open("json/unique_feature_values.json", "r") as file:
        unique_values_dict = json.load(file)

    for feature in unique_values_dict.keys():
        if feature == "sector":
            data["sector"] = data["sector"].apply(
                lambda x: x if x in unique_values_dict[feature] else "기타지역"
            )
        else:
            data = data[data[feature].isin(unique_values_dict[feature])]

    with open("json/code_to_category.json", "r") as file:
        category_to_code = json.load(file)

    def convert_category_to_code(category):
        return int(category_to_code.get(category, -1))

    data["sector"] = data["sector"].apply(convert_category_to_code)

    idx_feat = ["shippingitem_uuid"]
    common_feat_con = []
    common_feat_cat = ["sector"]
    trg_feat_con = ["lat", "lng"]
    trg_feat_cat = [
        "postal_digit1",
        "postal_digit2",
        "postal_digit3",
        "postal_digit4",
        "postal_digit5",
    ]

    if not opt_use_dist:
        feature = (
            idx_feat + common_feat_con + common_feat_cat + trg_feat_con + trg_feat_cat
        )

        df = data[feature]
        cols_to_convert = common_feat_cat + trg_feat_cat

        if use_cat:
            df[cols_to_convert] = df[cols_to_convert].astype("uint8")
        else:
            df[cols_to_convert] = df[cols_to_convert].astype("float32")
    elif opt_use_dist:
        feature = (
            idx_feat + common_feat_con + common_feat_cat + trg_feat_con + trg_feat_cat
        )

        df = data[feature]
        cols_to_convert = common_feat_cat + trg_feat_cat

        if use_cat:
            df[cols_to_convert] = df[cols_to_convert].astype("uint8")
        else:
            df[cols_to_convert] = df[cols_to_convert].astype("float32")
    else:
        raise ValueError

    feat_name_dict = {
        "idx_feat": idx_feat,
        "common_feat_con": common_feat_con,
        "common_feat_cat": common_feat_cat,
        "trg_feat_con": trg_feat_con,
        "trg_feat_cat": trg_feat_cat,
    }
    return df, feat_name_dict
