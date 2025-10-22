#!/bin/bash

# Detect OS for sed compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE="sed -i ''"
else
    SED_INPLACE="sed -i"
fi

git clone https://github.com/microsoft/TRELLIS.git
cd TRELLIS
git checkout 6b0d647
cd ..

mkdir -p atoken_inference/model/gs/representations/gaussian
mkdir -p atoken_inference/model/gs/renderers
mkdir -p atoken_inference/model/gs/utils

cp TRELLIS/trellis/representations/__init__.py atoken_inference/model/gs/representations/__init__.py
cp TRELLIS/trellis/representations/gaussian/__init__.py atoken_inference/model/gs/representations/gaussian/__init__.py
cp TRELLIS/trellis/representations/gaussian/gaussian_model.py atoken_inference/model/gs/representations/gaussian/gaussian_model.py
cp TRELLIS/trellis/representations/gaussian/general_utils.py atoken_inference/model/gs/representations/gaussian/general_utils.py

$SED_INPLACE '/^import utils3d$/d' atoken_inference/model/gs/representations/gaussian/gaussian_model.py
$SED_INPLACE 's/def save_ply(self, path, transform=\[\[1, 0, 0\], \[0, 0, -1\], \[0, 1, 0\]\])/def save_ply(self, path, transform=None)/' atoken_inference/model/gs/representations/gaussian/gaussian_model.py
$SED_INPLACE 's/def load_ply(self, path, transform=\[\[1, 0, 0\], \[0, 0, -1\], \[0, 1, 0\]\])/def load_ply(self, path, transform=None)/' atoken_inference/model/gs/representations/gaussian/gaussian_model.py
$SED_INPLACE '/if transform is not None:/,/rotation = utils3d\.numpy\.matrix_to_quaternion(rotation)/d' atoken_inference/model/gs/representations/gaussian/gaussian_model.py

cp TRELLIS/trellis/renderers/__init__.py atoken_inference/model/gs/renderers/__init__.py
cp TRELLIS/trellis/renderers/gaussian_render.py atoken_inference/model/gs/renderers/gaussian_render.py
cp TRELLIS/trellis/renderers/sh_utils.py atoken_inference/model/gs/renderers/sh_utils.py

cp TRELLIS/trellis/utils/__init__.py atoken_inference/model/gs/utils/__init__.py
cp TRELLIS/trellis/utils/general_utils.py atoken_inference/model/gs/utils/general_utils.py
cp TRELLIS/trellis/utils/random_utils.py atoken_inference/model/gs/utils/random_utils.py
cp TRELLIS/trellis/utils/random_utils.py atoken_inference/model/gs/random_utils.py
cp TRELLIS/trellis/utils/postprocessing_utils.py atoken_inference/model/gs/utils/postprocessing_utils.py
cp TRELLIS/trellis/utils/render_utils.py atoken_inference/model/gs/utils/render_utils.py

$SED_INPLACE '/"MeshRenderer": "mesh_renderer",/d' atoken_inference/model/gs/renderers/__init__.py
$SED_INPLACE '/from \.mesh_renderer import MeshRenderer/d' atoken_inference/model/gs/renderers/__init__.py
$SED_INPLACE '/from \.octree_renderer import OctreeRenderer/d' atoken_inference/model/gs/renderers/__init__.py

$SED_INPLACE '/from \.mesh import MeshExtractResult/d' atoken_inference/model/gs/representations/__init__.py
$SED_INPLACE '/from \.octree import DfsOctree as Octree/d' atoken_inference/model/gs/representations/__init__.py
$SED_INPLACE '/from \.radiance_field import Strivec/d' atoken_inference/model/gs/representations/__init__.py

$SED_INPLACE 's/from \.\.renderers import OctreeRenderer, GaussianRenderer, MeshRenderer/from ..renderers import GaussianRenderer/' atoken_inference/model/gs/utils/render_utils.py
$SED_INPLACE 's/from \.\.representations import Octree, Gaussian, MeshExtractResult/from ..representations import Gaussian/' atoken_inference/model/gs/utils/render_utils.py
$SED_INPLACE '/from \.\.modules import sparse as sp/d' atoken_inference/model/gs/utils/render_utils.py

$SED_INPLACE '/^def get_renderer(sample, \*\*kwargs):$/,/^def render_frames/{ /^def render_frames/!d; }' atoken_inference/model/gs/utils/render_utils.py

$SED_INPLACE '/^def render_frames/,/^    return rets$/{
s/    renderer = get_renderer(sample, \*\*options)/    if isinstance(sample, Gaussian) or "Gaussian" in str(type(sample)):\
        renderer = GaussianRenderer()\
        renderer.rendering_options.resolution = options.get("resolution", 512)\
        renderer.rendering_options.near = options.get("near", 0.8)\
        renderer.rendering_options.far = options.get("far", 1.6)\
        renderer.rendering_options.bg_color = options.get("bg_color", (0, 0, 0))\
        renderer.rendering_options.ssaa = options.get("ssaa", 1)\
        renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)\
        renderer.pipe.use_mip_gaussian = True\
        device = sample.device\
    else:\
        raise ValueError(f"Unsupported sample type: {type(sample)}")/
/^        if isinstance(sample, MeshExtractResult):$/,/^        else:$/d
s/^            /        /
}' atoken_inference/model/gs/utils/render_utils.py

$SED_INPLACE 's/from \.\.renderers import .*GaussianRenderer.*/from ..renderers import GaussianRenderer/' atoken_inference/model/gs/utils/postprocessing_utils.py
$SED_INPLACE 's/from \.\.representations import .*Gaussian.*/from ..representations import Gaussian/' atoken_inference/model/gs/utils/postprocessing_utils.py
$SED_INPLACE '/^def simplify_gs(/,/^def /{ /^def simplify_gs(/,/^    return new_gs$/d; }' atoken_inference/model/gs/utils/postprocessing_utils.py

rm -rf TRELLIS
