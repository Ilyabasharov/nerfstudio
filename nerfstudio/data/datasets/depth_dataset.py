# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Depth dataset.
"""

from typing import Dict, Union, List

import gc
import json
from pathlib import Path
import numpy as np
import numpy.typing as npt
import cv2
import torch
from torch import Tensor
from jaxtyping import Float
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.model_components import losses
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import torch_compile


class DepthDataset(InputDataset):
    """Dataset that returns images and depths.
    If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
        use_pseudodepth: Whether to generate monodepth data for supervision.
    """

    exclude_batch_keys_from_device: List[str] = InputDataset.exclude_batch_keys_from_device + ["depth_image", "depth_ranking_image"]

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        use_pseudodepth: bool = True,
        **kwargs,
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert len(dataparser_outputs.image_filenames) > 0, "Dataparser outputs are empty."
        # if there are no depth images than we want to generate them all with zoe depth
        # or we want to use both supervision
        self.generate_preudodepth = False

        if (
            "depth_filenames" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["depth_filenames"] is None
        ):
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            self.generate_preudodepth = True
            CONSOLE.print("[bold red] Using pseudodepth: forcing depth loss to be ranking loss.")
        elif use_pseudodepth:
            self.generate_preudodepth = True
            CONSOLE.print("[bold red] Using pseudodepth as an additional supervision.")

        if self.generate_preudodepth:
            cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
            # Note: this should probably be saved to disk as images, and then loaded with the dataparser.
            # That will allow multi-gpu training.
            if cache.exists():
                CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
                # load all the depths
                self.depths = torch.load(cache)
            else:
                self.depths = {}
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                transforms = self._find_transform(dataparser_outputs.image_filenames[0])
                data = dataparser_outputs.image_filenames[0].parent
                if transforms is not None:
                    with open(transforms, "r", encoding="utf-8") as file:
                        meta = json.load(file)
                    frames = meta["frames"]
                    filenames = [data / frames[j]["file_path"].split("/")[-1] for j in range(len(frames))]
                else:
                    meta = None
                    frames = None
                    filenames = dataparser_outputs.image_filenames

                pseudodepth_model = self._get_pseudodepth_model()

                for i in track(range(len(filenames)), description="Generating depth images"):
                    image_filename = filenames[i]
                    image: npt.NDArray[np.uint8] = cv2.imread(str(image_filename.absolute()))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # shape is (h, w) or (h, w, 3 or 4)
                    if self.scale_factor != 1.0:
                        height, width, _ = image.shape
                        newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                        image = cv2.resize(image, newsize, interpolation=cv2.INTER_AREA)
                    if len(image.shape) == 2:
                        image = image[:, :, None].repeat(3, axis=2)
                    
                    image = torch.from_numpy(image.astype("float32") / 255.0)

                    with torch.no_grad():
                        image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                        if image.shape[1] == 4:
                            image = image[:, :3, :, :]
                        depth_tensor = pseudodepth_model.infer(image).squeeze().unsqueeze(-1).cpu()

                    self.depths[image_filename] = depth_tensor

                torch.save(self.depths, cache)

                # clear unused memory
                del pseudodepth_model
                gc.collect()
                torch.cuda.empty_cache()

            if losses.FORCE_PSEUDODEPTH_LOSS:
                dataparser_outputs.metadata["depth_filenames"] = None
                dataparser_outputs.metadata["depth_unit_scale_factor"] = 1.0
                self.metadata["depth_filenames"] = None
                self.metadata["depth_unit_scale_factor"] = 1.0

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def _get_pseudodepth_model(
        self,
        **kwargs,
    ) -> torch.nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        repo = "isl-org/ZoeDepth"
        pseudodepth_model = torch_compile(
            torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device),
            mode="reduce-overhead",
        )

        return pseudodepth_model

    def get_metadata(self, data: Dict) -> Dict[str, Float[Tensor, "h w 1"]]:
        output_dict = {}
        if self.generate_preudodepth:
            output_dict["depth_ranking_image"] = self.depths[
                self._dataparser_outputs.image_filenames[data["image_idx"]]
            ]
        if self.depth_filenames is not None:
            filepath = self.depth_filenames[data["image_idx"]]
            height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
            width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

            # Scale depth images to meter units and also by scaling applied to cameras
            scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
            output_dict["depth_image"] = get_depth_image_from_path(
                filepath=filepath,
                height=height,
                width=width,
                scale_factor=scale_factor,
            )

        return output_dict

    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None
