import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches

'''
You only need to change the "num_random_samples", "direction_name" and "sigma_list" variables to test directions over
different shift values.

Please choose the direction from this list:
['age','bicolor','body_color','eye_color','rotation']
'''

direction_name = 'bicolor' # direction for testing
sigma_list = [-8,-4,0,4,8] # shifts along the direction
num_random_samples = 5 # number of random samples for the visualization

num_sigmas = len(sigma_list)
sigma_labels = np.arange(0,num_sigmas) - int(np.floor(num_sigmas/2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretained_model = "stylegan2-afhqv2-512x512.pkl"
with open(pretained_model, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()
num_ws = G.mapping.num_ws
data_dim = G.w_dim

style_dict = {'age':[0,2],
              'bicolor':[6,7],
              'body_color':[7,7],
              'eye_color':[12,12],
              'rotation':[2,3]
              }

direction = torch.from_numpy(np.load(f'{direction_name}.npy').reshape(1, data_dim)).to(device)
w1_idx = style_dict[direction_name][0]
w2_idx = style_dict[direction_name][1]

images_list = []

for i in range(num_random_samples):

    random_vector = torch.randn([1, data_dim]).to(device)
    w_vector = G.mapping(random_vector, c=None, truncation_psi=0.5, truncation_cutoff=None)
    w_vector_single = w_vector[0, 0:1].to(device)

    for sigma_idx in range(num_sigmas):

        end_point = w_vector.clone()
        end_point[0, w1_idx:(w2_idx + 1), :] = end_point[0,w1_idx:(w2_idx + 1), :] + (sigma_list[sigma_idx] * direction)
        img = G.synthesis(end_point, noise_mode='const', force_fp32=True)
        rgb_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_final = rgb_img[0].cpu().numpy()
        images_list.append(img_final)

fig = plt.figure(figsize=(16, 9))
grid = ImageGrid(fig, 111, nrows_ncols=(num_random_samples, num_sigmas), axes_pad=0.01,)

counter = 0
square_indices = [int(np.floor(num_sigmas/2)), int(np.floor(num_sigmas/2))+num_sigmas, int(np.floor(num_sigmas/2))+(2*num_sigmas)]
for ax, im in zip(grid, images_list):
    ax.imshow(im)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if counter in square_indices:
        rectangle = patches.Rectangle((0, 0), 511, 511, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)

    if np.remainder(counter, num_sigmas) == 0:
        counter = 0

    ax.annotate(f'{sigma_labels[counter]}$\sigma$', (20, 495), fontsize=16, color='white')
    counter += 1

plt.suptitle(f'Direction = {direction_name}')
plt.show()