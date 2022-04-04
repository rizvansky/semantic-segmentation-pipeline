from pipeline.registry import registry
from parser import Parser


if __name__ == "__main__":
    parser = Parser("./configs/train/baseline.yaml")
    parser.parse()
    print("x")
    print(registry["UnetDecoder"])
