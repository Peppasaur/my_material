
# Material-Capture Project

This project trains a neural BRDF model by rendering a sphere with known camera and lighting setups. We use a physically-based rendering model (PBR) as ground truth and try to match it using a neural network. The whole pipeline is differentiable, so we optimize the BRDF via gradient descent. 

---

## 1. Installation

Set up the environment with the following commands:

```bash
conda create --name material-capture python=3.10 pip
conda activate material-capture

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
conda install conda-forge::hydra-core
```

Make sure everything installs properly. After setup, you can run a simple dummy training.

---

## 2. Run the Training (Direct Illumination)

Run:

```bash
python main.py renderer=dynamicpoint_emitter outfolder=YOUR_OUTPUT_PATH
```

This will start a simple training process using direct lighting:

- The object is a unit sphere.
- Cameras are sampled on a sphere surface.
- Each iteration dynamically generates 4 point lights.
- The ground truth image is rendered using a microfacet PBR model.

After training:

- Rendered output is saved to:  
  `$outfolder/sphere/$experiment_name/roughness_0.20_metallic_0.20`
- Checkpoints are stored in:  
  `$outfolder/sphere/$experiment_name/training`
- Logs are uploaded via PyTorch Lightning.

---

## 3. What You Need to Do

### A. Implement Camera Functions  
File: `util/dataset/sphere.py`

You need to finish camera setup for the synthetic scene.

- `get_camera_dicts(self)`:  
  Generate a list of camera dictionaries by uniformly sampling the viewing directions over a sphere.

- `get_ray_directions(H, W, focal)`:  
  Calculate ray directions from the camera center. Implement a pinhole projection.

- `get_c2w(camera)`:  
  Construct a proper camera-to-world transformation matrix from position, look_at, and up vector.

---

### B. Implement the PBR BRDF  
File: `model/brdf.py`, class: `PBRBRDF`

Implement the following function:

```python
def eval_brdf(self, wi, wo, normal)
```

This is the PBR model based on microfacet theory. The formula is:

```math
f(\mathbf{x}, \omega_i, \omega_o) = \frac{\mathbf{k}_d(\mathbf{x})}{\pi} \left( \mathbf{n} \cdot \omega_i \right)_+ + \frac{F(\omega_i, \mathbf{h}, \mathbf{k}_s(\mathbf{x})) \, D(\mathbf{h}, \mathbf{n}, \sigma(\mathbf{x})) \, G(\omega_i, \omega_o, \mathbf{n}, \sigma(\mathbf{x}))}{4 (\mathbf{n} \cdot \omega_o)}
```

**Where:**

- `ω_i`: incoming light direction  
- `ω_o`: view direction  
- `n`: surface normal  
- `h = (ω_i + ω_o) / ||ω_i + ω_o||`: half vector  
- `σ(x)`: roughness  
- `k_d(x)`: diffuse albedo  
- `k_s(x)`: specular color

**Use the utility functions in** `utils.ops`:
- `D_GGX(...)`  
- `G_Smith(...)`  
- `fresnelSchlick(...)`

#### Material Parameter Note

From the material properties (`a` for albedo, `m` for metallic), compute:

- `k_d = a * (1 - m)`  
- `k_s = 0.04 * (1 - m) + a * m`

---

### C. Finish the Neural BRDF Model  
File: `model/brdf.py`, class: `MLPBRDF`

This is the learnable model. Complete the TODOs inside the class, especially the MLP structure and the encoding. You can tweak the architecture and optimization configs in:

- `config/model/base.yaml`
- `config/data/base.yaml`

After training, rendered outputs will be saved in the same format as the PBR model.

---

## 4. Test with Environment Map

After finishing the neural model, you can test it under an environment map:

```bash
python test.py renderer=envmap_emitter ckpt_path=PATH_TO_CHECKPOINT
```

You’ll need to implement the following function in `model/emitter.py`:

- `eval_emitter(position, light_dir)`:  
  Return radiance and PDF for each light direction based on the environment map.

```python
Returns:
    Le: Bx3 radiance
    pdf: Bx1 PDF
    valid: B valid sample mask (always True)
```

---

## 5. Rendering and Video Output

Once test.py is done, the rendered frames will be saved and you can make a video from them. The camera follows a circular path around the sphere, controlled by:

```yaml
cfg.renderer.camera.number_of_views
```
