import argparse
import torch
from safetensors.torch import load_file

# 参数解析
parser = argparse.ArgumentParser(description='Convert between SafeTensors and PyTorch Tensors')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='the input file path')
parser.add_argument('--output', '-o', type=str, required=True,
                    help='the output file path')

if __name__ == "__main__":
    # 解析参数
    args = parser.parse_args()

    x = load_file(args.input)
    for tensor_name in x.keys():
        print(tensor_name)

    # 保存 PyTorch Tensor
    torch.save(x, args.output)
