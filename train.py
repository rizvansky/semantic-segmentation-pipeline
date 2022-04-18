from parser import Parser


if __name__ == "__main__":
    parser = Parser("./configs/train/baseline.yaml")
    pipeline = parser.parse()
    pipeline.run()

