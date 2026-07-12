from data import generate_dataset


def main():
    x, y = generate_dataset(10)

    print("X Values:")
    print(x)

    print("\nY Values:")
    print(y)


if __name__ == "__main__":
    main()