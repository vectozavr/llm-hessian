import torch
from utils import check_gpus, plot_heatmap, plot_hist


def main():
    hess = torch.load("data/hessian_diag_q_proj_hutchinson_samples_10000.pt")
    plot_hist(torch.abs(hess[0, :50]))


if __name__ == '__main__':
    check_gpus()
    main()
