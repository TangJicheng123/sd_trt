import argparse
from safetensors import safe_open


def print_tensor_names(filename):
    # 加载 SafeTensors 文件
    safe_tensors = safe_open(filename, framework="pt", device="cpu")

    # 打印每个 Tensor 的名称
    for tensor_name in safe_tensors.keys():
        print(tensor_name)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Print tensor names from a SafeTensors file')
    parser.add_argument('--input', '-i', type=str, help='path to the SafeTensors file')
    args = parser.parse_args()

    # 调用函数打印 Tensor 名称
    print_tensor_names(args.input)
