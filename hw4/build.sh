set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        git

    pip3 install -qq \
        albumentations==1.3.1 \
        imageio==2.31.5 \
        joblib \
        matplotlib==3.8.1 \
        opencv-python==4.8.1.78 \
        pandas==2.1.3 \
        pillow==10.1.0 \
        pytorch-msssim==1.0.0 \
        gdown
    mkdir /weights
    gdown -O /weights/p1q2.pth 1ALDFo87IJrKzGck2MEGymsgzsczh4ZkG
    chmod -R 0777 /weights
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"