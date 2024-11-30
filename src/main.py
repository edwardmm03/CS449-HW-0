import tensorflow as tf
import numpy as np
from argparse import ArgumentParser, Namespace
from get_file import get_file
from model import Model

INTEGER_CHARS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"}


def create_dataset(
    data: np.ndarray[np.ndarray[np.float32]],
    labels: np.ndarray[np.float32],
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((data, labels))


def parse_data(line: str) -> np.ndarray[np.float32]:
    raw_data: list[str] = line.split(" ")
    data: list[np.float32] = []
    for point in raw_data:
        if point != "\n":
            data.append(np.float32(point))
    return np.array(data, dtype=np.float32)


def get_data(file_path: str) -> np.ndarray[np.ndarray[np.float32]]:
    with get_file(file_path).open("r") as file:
        lines = file.readlines()
    data: list[np.ndarray[np.float32]] = []
    for line in lines:
        parsed_line = parse_data(line)
        if len(parsed_line) > 0:
            data.append(parsed_line)
    return np.array(data, dtype=np.float32)


def parse_label(raw_label: str) -> np.float32:
    """takes a string containing a numerical value
    strips all non numeric characters and outputs the integer value"""
    label: str = ""
    for c in raw_label:
        if c == ".":
            break
        if c in INTEGER_CHARS:
            label += c
    return np.float32(label)


def convert_labels(labels: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    converted_labels = np.zeros(len(labels), dtype=np.float32)
    for i in range(len(labels)):
        converted_labels[i] = (labels[i] + 1) / 2
    return converted_labels


def revert_labels(labels: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    reverted_labels = np.zeros(len(labels), dtype=np.float32)
    for i in range(len(labels)):
        reverted_labels[i] = (labels[i] * 2) - 1
    return reverted_labels


def get_labels(file_path: str) -> np.ndarray[np.float32]:
    with get_file(file_path).open("r") as file:
        raw_labels: list[str] = file.read().split(",")
    labels: np.ndarray[np.float32] = np.zeros(
        len(raw_labels), dtype=np.float32
    )
    for i in range(len(labels)):
        labels[i] = parse_label(raw_labels[i])
    return convert_labels(labels)


def loss(
    prediction: np.ndarray[np.float32], expected: np.ndarray[np.float32]
) -> np.float32:
    return np.mean(np.absolute(prediction - expected))


if __name__ == "__main__":
    arg_parser: ArgumentParser = ArgumentParser()
    arg_parser.add_argument(
        "--train_data",
        default="../data/train/a2-train-data.txt",
        help="path to training data file",
    )
    arg_parser.add_argument(
        "--train_label",
        default="../data/train/a2-train-label.txt",
        help="path to training data label file",
    )
    arg_parser.add_argument(
        "--test_data",
        default="../data/test/a2-test-data.txt",
        help="path to test data file",
    )
    arg_parser.add_argument(
        "--test_label",
        default="../data/test/a2-test-label.txt",
        help="path to test data label file",
    )
    arg_parser.add_argument(
        "--network_file",
        "-n",
        default="../saves/model.keras",
        help="network to load from save",
    )
    args: Namespace = arg_parser.parse_args()

    train_data: tf.data.Dataset = create_dataset(
        get_data(args.train_data), get_labels(args.train_label)
    )

    model: Model = Model()
    model.new(1000, 2)
    model.fit(train_data, 100, 10)
    results: np.ndarray[np.float32] = revert_labels(
        model.predict(get_data(args.test_data))
    )
    print(loss(results, get_labels(args.test_label)))
    print(results)
    f = open("output.txt","w")
    for r in results:
        cur = str(r)
        f.write(cur)
        f.write(" ")
    f.close()