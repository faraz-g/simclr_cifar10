import torch
from data_loader import get_cifar_dataset
from torch.utils.data import DataLoader
import os
from model import SimCLR
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

BATCH_SIZE = 512
MODEL_PATH = "checkpoints/model_epoch_499_512bs"


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []

    for x, y in loader:
        x = x.to(device)

        with torch.no_grad():
            h = simclr_model.feature_extractor(x)

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = get_cifar_dataset(
        root_folder="/home/faraz/Documents/code/simclr_cifar10/cifar10", train=True, aug=False
    )

    test_dataset = get_cifar_dataset(
        root_folder="/home/faraz/Documents/code/simclr_cifar10/cifar10", train=False, aug=False
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, drop_last=True
    )

    trained_simclr_model = SimCLR()
    trained_simclr_model.load_state_dict(torch.load(MODEL_PATH))
    trained_simclr_model = trained_simclr_model.to(device)

    (X_train, y_train, X_test, y_test) = get_features(
        trained_simclr_model, train_dataloader, test_dataloader, device=device
    )
    print(f"X_train shape {X_train.shape}")
    print(f"y_train shape {y_train.shape}")
    print(f"X_test shape {X_test.shape}")
    print(f"y_test shape {y_test.shape}")

    lr1 = LogisticRegression(penalty="l2", solver="saga", multi_class="auto", max_iter=100)
    lr1.fit(X_train, y_train)

    y_pred = lr1.predict(X_test)

    print(f"Accuracy with SimCLR Representations: {accuracy_score(y_test, y_pred)}")

    print("Testing PCA Baseline...")
    lr2 = LogisticRegression(penalty="l2", solver="saga", multi_class="auto", max_iter=100)

    X_train, y_train = train_dataset.data, np.asarray(train_dataset.targets)
    X_test, y_test = test_dataset.data, np.asarray(test_dataset.targets)

    print(f"X_train shape {X_train.shape}")
    print(f"y_train shape {y_train.shape}")
    print(f"X_test shape {X_test.shape}")
    print(f"y_test shape {y_test.shape}")

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 3072)
    X_test = X_test.reshape(-1, 3072)

    pca = PCA(n_components=512)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"X_train_pca shape {X_train_pca.shape}")
    print(f"X_test_pca shape {X_test_pca.shape}")

    lr2.fit(X_train_pca, y_train)
    y_pred_pca = lr2.predict(X_test_pca)

    print(f"Accuracy with PCA Representations: {accuracy_score(y_test, y_pred_pca)}")


if __name__ == "__main__":
    main()
