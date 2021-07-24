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
download and uncompress the `esrgan-tf2_1.tar.gz` model on
https://tfhub.dev/captain-pool/esrgan-tf2/1

## run
On Window, uncomment this line and check the correct CUDA path
```py
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
```

```sh
python3 nearsightlens.py
```

## Reference
https://www.tensorflow.org/hub/tutorials/image_enhancing