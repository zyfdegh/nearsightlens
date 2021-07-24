Super resolution with TensorFlow for image enhancing.

## macOS

My running machine

* macOS 11.4
* python 3.9.5
* pip 21.1.1

* tensorflow 2.5.0
* numpy 1.19.5
* cuda (optional)

```sh
$ pip3 install tensorflow \
    tensorflow-hub \
    matplotlib \
    keras \
    numpy \
    matplotlib
```

## model
Download and decompress the `esrgan-tf2_1.tar.gz` model from
https://tfhub.dev/captain-pool/esrgan-tf2/1

## run
On Window, uncomment this line to load CUDA library (use the correct path)
```py
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
```

```sh
python3 nearsightlens.py
```

## Troubleshoot
1. Allocation Memory: Allocation of 38535168 exceeds 10% of system memory

Solution: Crop your original.png to smaller, e.g. 320x320

## Reference
https://www.tensorflow.org/hub/tutorials/image_enhancing