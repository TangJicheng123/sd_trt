wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2022.05-Linux-x86_64.sh

conda create -y -n sd_trt python=3.10
conda activate sd_trt
pip install -r requirements.txt