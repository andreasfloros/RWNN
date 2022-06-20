# RWNN
Andreas Floros, *Imperial College London, United Kingdom*

Abstract
----------
Image Restoration (IR) consists of a family of ill-posed inverse problems which have been studied extensively
through the lens of model and learning-based methods. Recent advances combining the two approaches have
allowed denoisers to solve general IR problems, thus reducing the complexity of the tasks to just denoising. In
this work, a multipurpose image processing tool is developed based on wavelet theory and Sparse Coding (SC).
The proposed method exploits the recursive nature of wavelet transforms to achieve high generalisation ability
and provides a powerful denoising engine. The utility of the proposed extends beyond traditional Additive
White Gaussian Noise (AWGN) removal; this flexibility is demonstrated in compact representation of images,
progressive loading, spatially variant noise removal, deblurring and inpainting.

|<img align="center" src="examples/castle/castlen.png" width="240px"/> | <img align="center" src="examples/castle/castlentrans.png" width="240px"/> | <img align="center" src="examples/castle/castlerestrans.png" width="240px"/> | <img align="center" src="examples/castle/castleres.png" width="240px"/>|
|:---:|:---:|:---:|:---:|
|<i>(a) Noisy image, σ=25</i>|<i>(b) Noisy transformed</i>|<i>(c) Denoised transformed</i>|<i>(d) Denoised</i>|

Installation
----------
* Create a virtual environment: `python -m venv env`
* Activate venv: `env\Scripts\activate`
* Install requirements: `pip install -r requirements.txt`


Training from scratch
----------
RWNN is trained in two steps, for the default settings proceed as follows:
* First run `python train.py --should_prepare True` to train in the Denoising AutoEncoder (DAE) setting
* Run `python train.py --dae False --epoch_start E` to fine-tune epoch E for the final RWNN-F model

To explore hyperparameter settings use the `-h` flag. Inspect the `models` folder if you wish to tweak the networks.

Testing
----------
Pretrained RWNN-DAE and RWNN-F are found in the `logs` folder (epoch 41 and 73 respectively).
* `python test.py -h` for testing the networks in the DAE and denoising tasks.
* `python deblur.py -h` and `python inpaint.py -h` for Plug-and-Play Prior (P3) deblurring and inpainting respectively.

The default data for testing is Set12 and the results are included in the `examples` folder.
