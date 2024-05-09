set -o xtrace

setup_root() {
    apt-get install -qq -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-pip \
        python3-tk \
        git

    pip3 install -qq \
        pytest \
        scikit-image==0.19.3 \
        scikit-learn==1.1.2 \
        opencv-python \
        matplotlib \
        imgaug \
        timm==0.6.11 \
        moviepy
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"