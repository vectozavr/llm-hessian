import torch
from utils import check_gpus, plot_heatmap, plot_hist


def main():
    #hess = torch.load("data/diag_hessian/hessian_diag_q_proj_vhp_samples_1.pt")

    vhp_samples_list = [1, 10, 100, 1000, 3000, 5000]
    errors = []

    for i in range(len(vhp_samples_list)-1):

        hess1 = torch.load("data/diag_hessian/hessian_diag_q_proj_vhp_samples_" + str(vhp_samples_list[i]) + ".pt")
        hess2 = torch.load("data/diag_hessian/hessian_diag_q_proj_vhp_samples_" + str(vhp_samples_list[i+1]) + ".pt")

        errors.append(torch.linalg.norm(hess2 - hess1))


    #plot_hist(torch.abs(hess[0, :50]))
    #plot_heatmap(torch.abs(hess))


if __name__ == '__main__':
    check_gpus()
    main()
