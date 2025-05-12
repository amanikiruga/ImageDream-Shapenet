import json 
import glob
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from .shared_dataset import SharedDataset
ROOT= os.getenv("SPLATTER_IMAGE_ROOT")
SHAPENET_MODIFIED_ROOT = os.getenv("SHAPENET_MODIFIED_CHAIRS_ROOT")
CHAIRS_TRAIN_TEST_SPLIT_PATH = f"{ROOT}/shapenet_chairs_split.json"

assert SHAPENET_MODIFIED_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

class ShapenetChairsModified(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train", deterministic_test_idxs = None, override_example_ids = None, override_overall_view = None, additional_overall_views=None, save_splits = False, only_poses= False, override_num_input_images=None):
        super().__init__()
        self.cfg = cfg

        self.num_input_images = override_num_input_images or cfg.data.input_images
        self.only_poses = only_poses
        self.additional_overall_views = additional_overall_views or []
        
        self.deterministic_test_idxs = deterministic_test_idxs
        self.is_val_in_dist = (dataset_name == "val_in_dist")
        self.is_val_ood = (dataset_name == "val_ood")
        self.dataset_name = dataset_name
        if dataset_name == "vis" or dataset_name == "val_ood" or dataset_name == "val_in_dist":
            self.dataset_name = "test"
            
        # self.base_path = os.path.join(SHAPENET_MODIFIED_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
        #                                                                            cfg.data.category,
        #                                                                            self.dataset_name))
        # self.base_path = os.path.join(SHAPENET_MODIFIED_ROOT, "02958343") # use if using /nobackup/nvme1/ShapeNetCore.v2.modified/
        self.base_path = SHAPENET_MODIFIED_ROOT

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp
                
        overall_view = cfg.data.overall_view
        if override_overall_view is not None:
            overall_view = override_overall_view

        if cfg.data.overall_view_two: 
            self.additional_overall_views.append(cfg.data.overall_view_two)
            print("WARNING: using two overall views", cfg.data.overall_view, cfg.data.overall_view_two) 
        

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", overall_view, "intrinsics.txt"))
        )

        # self.intrins = [x for x in self.intrins if len(glob.glob(os.path.join(os.path.dirname(x), "rgb", "*"))) == 24 and len(glob.glob(os.path.join(os.path.dirname(x), "pose", "*"))) == 24]
        self.intrins = [x for x in self.intrins if len(glob.glob(os.path.join(os.path.dirname(x), "rgb", "*"))) > 0 and len(glob.glob(os.path.join(os.path.dirname(x), "pose", "*"))) > 0]
        # Split the data into train and test sets
        
        with open(CHAIRS_TRAIN_TEST_SPLIT_PATH, "r") as f:
            object_splits = json.load(f)
        
        self.train_intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) in object_splits["train"]]
        self.test_intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) in object_splits["test"]]
        
        if self.dataset_name == "train":
            self.intrins = self.train_intrins
        elif self.is_val_ood or self.is_val_in_dist:
            # shuffle test_intrins and take the first cfg.data.val_size
            # seed 
            np.random.seed(42)
            np.random.shuffle(self.test_intrins)
            self.intrins = self.test_intrins[:cfg.data.val_size]
            # print example ids of self.intrins
            print("example ids of val_ood or val_in_dist dataset", [os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) for intrin_path in self.intrins])
            assert len(self.intrins) > 0, "size of val_ood or val_in_dist dataset is 0"
        else: 
            np.random.seed(42)
            np.random.shuffle(self.test_intrins)
            self.intrins = self.test_intrins[:cfg.data.val_size]
            
        if override_example_ids is not None:
            self.intrins = [intrin_path for intrin_path in self.intrins if os.path.basename(os.path.dirname(os.path.dirname(intrin_path))) in override_example_ids]
            print("overriding example ids", override_example_ids, "length of intrins:", len(self.intrins), "self.intrins:", self.intrins)
            
            

        print("length of intrinsics", len(self.intrins))
        if cfg.data.subset != -1:
            self.intrins = self.intrins[:cfg.data.subset]

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj
   
            
        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.num_input_images == 1:
            # self.test_input_idxs = [64]
            if deterministic_test_idxs is not None:
                assert len(deterministic_test_idxs) == 1
                self.test_input_idxs = torch.tensor(deterministic_test_idxs)
                print("using deterministic test idxs:", self.test_input_idxs)
            else: 
                self.test_input_idxs = torch.randint(0, 24, (cfg.data.val_size,))
                print("Chosen random testing input idxs", self.test_input_idxs)
        else:
            # self.test_input_idxs = [64, 128]
            self.test_input_idxs = torch.linspace(0, 47, self.num_input_images).long()
            print(f"self.num_input_images {self.num_input_images} is not 1")
            print("using linspace created test idxs:", self.test_input_idxs)
            

    def __len__(self):
        return len(self.intrins)

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        dir_path = os.path.dirname(intrin_path)
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        if self.only_poses:
            rgb_paths = [""] * len(pose_paths)  # dummy paths
        else: 
            rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
            
        assert len(rgb_paths) == len(pose_paths)
        
        for additional_view in self.additional_overall_views:
            additional_dir_path = os.path.join(os.path.dirname(dir_path), additional_view)
            rgb_paths_view = sorted(glob.glob(os.path.join(additional_dir_path, "rgb", "*")))
            pose_paths_view = sorted(glob.glob(os.path.join(additional_dir_path, "pose", "*")))
            assert len(rgb_paths_view) == len(pose_paths_view) and len(rgb_paths_view) == 24, \
                f"View paths length mismatch for view '{additional_view}: len(pose_paths_view): {len(rgb_paths_view)} !=  len(pose_paths_view): {len(pose_paths_view)} != 24'"
            rgb_paths += rgb_paths_view
            pose_paths += pose_paths_view

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []

            cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))], no_imgs=self.only_poses)

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T
                if self.only_poses: 
                    self.all_rgbs[example_id].append(torch.empty(0)) 
                else: 
                    self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)
            
            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]

        example_id = os.path.basename(os.path.dirname(os.path.dirname(intrin_path)))
        # print("intrin_path", intrin_path)
        
        if self.dataset_name == "test":
            split_to_print = "val_ood" if self.is_val_ood else "val_in_dist" if self.is_val_in_dist else "test"
            # print("example_id", example_id, "split", split_to_print, "overall_view", self.cfg.data.overall_view)

        self.load_example_id(example_id, intrin_path)
        if self.dataset_name == "train":
            if self.num_input_images == 1: 
                frame_idxs = torch.randperm(
                        len(self.all_rgbs[example_id])
                        )[:self.imgs_per_obj]

                frame_idxs = torch.cat([frame_idxs[:self.num_input_images], frame_idxs], dim=0)
            else: 
                input_idxs = self.test_input_idxs
                assert len(input_idxs) == self.num_input_images, f"input_idxs: {input_idxs} should be of length {self.num_input_images}"
                # deterministic with linspace depending on the number of images
                frame_idxs = torch.randperm(
                        len(self.all_rgbs[example_id])
                        )[:self.imgs_per_obj]
                frame_idxs = torch.cat([input_idxs, frame_idxs], dim=0)
                frame_idxs = frame_idxs.type(torch.int)

        else:
            # print("testing, input_idxs:", self.test_input_idxs)
            input_idxs = self.test_input_idxs
            
            # frame_idxs = torch.cat([torch.tensor(input_idxs), 
            #                         torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 
            if self.deterministic_test_idxs is not None:
                frame_idxs = torch.cat([input_idxs, 
                                        torch.tensor([i for i in range(len(self.all_rgbs[example_id])) if i not in input_idxs])], dim=0)
            else: 
                frame_idxs = torch.cat([input_idxs[index:index+1], 
                                        torch.tensor([i for i in range(len(self.all_rgbs[example_id])) if i not in input_idxs])], dim=0)
        try:         
            # print("example_id", example_id, "len of self.all_rgbs[example_id]", len(self.all_rgbs[example_id]))
            # print("frame_idxs", frame_idxs) 
            images_and_camera_poses = {
                "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
                "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
                "world_view_transforms_absolute": self.all_world_view_transforms[example_id][frame_idxs],
                "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
                "view_to_world_transforms_absolute": self.all_view_to_world_transforms[example_id][frame_idxs],
                "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
                "full_proj_transforms_absolute": self.all_full_proj_transforms[example_id][frame_idxs],
                "camera_centers": self.all_camera_centers[example_id][frame_idxs], 
                "camera_centers_absolute": self.all_camera_centers[example_id][frame_idxs],
                "example_id": example_id,
            }

            images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
            images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
        except Exception as e: 
            print("error in getting item")
            print("frame_idxs:", frame_idxs)
            raise e
        return images_and_camera_poses