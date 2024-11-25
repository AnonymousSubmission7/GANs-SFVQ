import numpy as np
import torch
import pickle
from colat.models.conditional import LinearConditional
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

'''
You only need to change "direction_name" and "sigma_list" variables to test directions over different shift values.
'''
# Please choose the direction from this list ['rotation', 'smile', 'hair_color', 'gender', 'age', 'bald', 'race']
direction_name = 'smile'           # direction for testing
sigma_list = [-8, -4, 0, 4, 8]     # shifts along the direction

num_sigmas = len(sigma_list)
sigma_labels = np.arange(0,num_sigmas) - int(np.floor(num_sigmas/2))
loading_address_sfvq = './directions_sfvq/'
loading_address_sefa = './directions_sefa/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretained_model = "stylegan2-ffhq-1024x1024.pkl"
with open(pretained_model, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()
num_ws = G.mapping.num_ws
data_dim = G.w_dim

directions_dict_sfvq = {'age':[3,7],
                  'smile':[4,4],
                  'gender':[4,7],
                  'hair_color':[8,8],
                  'bald':[3,6],
                  'rotation':[0,2],
                  'race':[8,9]}

directions_dict_sefa = {'age':[3,4],
                   'gender':[4,6],
                   'rotation':[0,2],
                   'race':[6,11],
                   'bald':[4,5],
                   'smile':[4,5],
                   'hair_color':[7,8]}

directions_dict_ganspace = {'gender': [[0], [0], [17]],
                       'rotation': [[1], [0], [2]],
                       'smile': [[43], [6], [7]],
                       'hair_color': [[10], [7], [8]]}

directions_dict_latentclr = {'rotation': [9,0,1],
                   'age': [14,6,13],
                   'smile':[28,4,5],
                   'hair_color': [2,6,13],
                    'bald':[48,2,5]}

methods_dict = {'rotation': ['SeFa', 'GANSpace', 'LatentCLR', 'SFVQ (ours)'],
                'smile': ['SeFa', 'GANSpace', 'LatentCLR', 'SFVQ (ours)'],
                'hair_color': ['SeFa', 'GANSpace', 'LatentCLR', 'SFVQ (ours)'],
                'age': ['SeFa', 'LatentCLR', 'SFVQ (ours)'],
                'bald': ['SeFa', 'LatentCLR', 'SFVQ (ours)'],
                'gender': ['SeFa', 'GANSpace', 'SFVQ (ours)'],
                'race':['SeFa', 'SFVQ (ours)']}


methods_list = methods_dict[direction_name]
num_methods = len(methods_list)

random_vector = torch.randn([1, data_dim]).to(device)
w_vector = G.mapping(random_vector, c=None, truncation_psi=0.5, truncation_cutoff=None)
w_vector_single = w_vector[0,0:1].to(device)

# SFVQ (Ours)
dir_sfvq = torch.from_numpy(np.load(loading_address_sfvq + f'{direction_name}.npy').reshape(1,data_dim)).to(device)
dir_cfgs_sfvq = directions_dict_sfvq[direction_name]

# SeFa
dir_sefa = torch.from_numpy(np.load(loading_address_sefa + f'{direction_name}.npy')).to(device)
dir_cfgs_sefa = directions_dict_sfvq[direction_name]

# GANSpace
if 'GANSpace' in methods_list:
    pca_comps = torch.load('ganspace_z_comp.pt')
    dir_cfgs_ganspace = directions_dict_ganspace[direction_name]
    if direction_name == 'gender':
        dir_ganspace = -1 * pca_comps[dir_cfgs_ganspace[0]][0].to(device)
    else:
        dir_ganspace = pca_comps[dir_cfgs_ganspace[0]][0].to(device)

# LatentCLR
if 'LatentCLR' in methods_list:
    dir_cfgs_latentclr = directions_dict_latentclr[direction_name]
    dir_idx_latentclr = dir_cfgs_latentclr[0]
    latent_clr_model = LinearConditional(k=100, size=data_dim)
    latent_clr_model = latent_clr_model.to(device)

if 'LatentCLR' in methods_list:
    dir_latentclr = latent_clr_model.nets[dir_idx_latentclr](w_vector_single)
    dir_latentclr = dir_latentclr / torch.linalg.norm(dir_latentclr)


images_list_sefa = []
images_list_gansapce = []
images_list_latentclr = []
images_list_sfvq = []

for sigma_idx in range(len(sigma_list)):

            end_point_sefa = w_vector.clone()
            end_point_gansapce = w_vector.clone()
            end_point_latentclr = w_vector.clone()
            end_point_sfvq = w_vector.clone()

            end_point_sfvq[0, dir_cfgs_sfvq[0]:(dir_cfgs_sfvq[1] + 1), :] = end_point_sfvq[0,dir_cfgs_sfvq[0]:(dir_cfgs_sfvq[1] + 1),:] + (sigma_list[sigma_idx] * dir_sfvq)
            img = G.synthesis(end_point_sfvq, noise_mode='const', force_fp32=True)
            rgb_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_final = rgb_img[0].cpu().numpy()
            images_list_sfvq.append(img_final)

            end_point_sefa[0, dir_cfgs_sefa[0]:(dir_cfgs_sefa[1] + 1), :] = end_point_sefa[0,dir_cfgs_sefa[0]:(dir_cfgs_sefa[1] + 1),:] + (sigma_list[sigma_idx] * dir_sefa)
            img = G.synthesis(end_point_sefa, noise_mode='const', force_fp32=True)
            rgb_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_final = rgb_img[0].cpu().numpy()
            images_list_sefa.append(img_final)

            if 'GANSpace' in methods_list:
                end_point_gansapce[0, dir_cfgs_ganspace[1][0]:(dir_cfgs_ganspace[2][0] + 1), :] = end_point_gansapce[0, dir_cfgs_ganspace[1][0]:(dir_cfgs_ganspace[2][0] + 1), :] + (sigma_list[sigma_idx] * dir_ganspace)
                img = G.synthesis(end_point_gansapce, noise_mode='const', force_fp32=True)
                rgb_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_final = rgb_img[0].cpu().numpy()
                images_list_gansapce.append(img_final)

            if 'LatentCLR' in methods_list:
                end_point_latentclr[0, dir_cfgs_latentclr[1]:(dir_cfgs_latentclr[2] + 1), :] = end_point_latentclr[0,dir_cfgs_latentclr[1]:(dir_cfgs_latentclr[2] + 1),:] + (sigma_list[sigma_idx] * dir_latentclr)
                img = G.synthesis(end_point_latentclr, noise_mode='const', force_fp32=True)
                rgb_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_final = rgb_img[0].cpu().numpy()
                images_list_latentclr.append(img_final)


images_list = images_list_sefa + images_list_gansapce + images_list_latentclr + images_list_sfvq

fig = plt.figure(figsize=(16, 9))
grid = ImageGrid(fig, 111, nrows_ncols=(num_methods, num_sigmas), axes_pad=0.01,)

counter = 0
xlabel_counter = 0
for ax, im in zip(grid, images_list):
    ax.imshow(im)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if np.remainder(counter, num_sigmas) == 0:
        ax.set_ylabel(f'{methods_list[xlabel_counter]}', fontsize=16)
        xlabel_counter += 1
        counter = 0

    ax.annotate(f'{sigma_labels[counter]}$\sigma$', (20, 1005), fontsize=16, color='white')
    counter += 1

plt.suptitle(f'Direction = {direction_name}')
plt.show()