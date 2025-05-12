import torch
from torch.utils.data import Dataset
from utils.graphics_utils import get_ray_directions, get_rays, get_mvp_matrix



class ShapenetChairsAsRaysDataset(Dataset):
    def __init__(self, shapenet_dataset):
        self.ds = shapenet_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        imgs = item["gt_images"]                    # [N,3,H,W]
        c2w = item["view_to_world_transforms"]      # [N,4,4]
        full_proj = item["full_proj_transforms"]    # [N,4,4]
        centers = item["camera_centers"]            # [N,3]
        Ks = item["intrinsics"]                     # you’ll need to add this key!

        N, _, H, W = imgs.shape

        # build directions per view using actual K
        directions_list = []
        for i in range(N):
            K = Ks[i][:3,:3]   # if you stored the 4×4, take the 3×3 part
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            dirs = get_ray_directions(H, W, (fx, fy), (cx, cy), use_pixel_centers=False)
            directions_list.append(dirs)
        directions = torch.stack(directions_list, dim=0)  # [N,H,W,3]

        # compute rays
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)  # each [N,H,W,3]

        # compute MVP
        mvp = get_mvp_matrix(c2w, full_proj)  # [N,4,4]

        return {
            "rays_o": rays_o,             # [N,H,W,3]
            "rays_d": rays_d,             # [N,H,W,3]
            "mvp_mtx": mvp,               # [N,4,4]
            "c2w": c2w,                   # [N,4,4]
            "camera_positions": centers,  # [N,3]
            "light_positions": centers,   # or zeros, as you prefer
            "gt_rgb": imgs,               # [N,3,H,W]
            "height": H,
            "width": W,
        }
