"""
Test the fields
"""
import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.utils.external import TCNN_EXISTS, tcnn_import_exception
from nerfstudio.cameras.bundle_adjustment import HashBundleAdjustment


def test_nerfacto_field():
    """Test the Nerfacto field"""
    if not TCNN_EXISTS:
        # tinycudann module doesn't exist
        print(tcnn_import_exception)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aabb_scale = 1.0
    aabb = torch.tensor(
        [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]],
        dtype=torch.float32,
        device=device,
    )
    field = NerfactoField(
        aabb=aabb,
        bundle_adjustment=HashBundleAdjustment(use_bundle_adjust=False),
        num_images=1,
    ).to(device)

    num_rays = 1024
    num_samples = 256
    positions = torch.rand((num_rays, num_samples, 3), dtype=torch.float32, device=device)
    directions = torch.rand_like(positions)
    bs = (*directions.shape[:-1], 1)
    frustums = Frustums(
        origins=positions,
        directions=directions,
        starts=torch.zeros(bs, device=device),
        ends=torch.zeros(bs, device=device),
        radii=torch.ones(bs, device=device),
    )
    ray_samples = RaySamples(
        frustums=frustums,
        camera_indices=torch.zeros(
            (num_rays, 1, 1),
            device=device,
            dtype=torch.int32,
        ),
    )
    field.forward(ray_samples)


if __name__ == "__main__":
    test_nerfacto_field()
