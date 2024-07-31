file_path = "F:/Git/vscodeAI/AI/Data/emnist-balanced-train-images-idx3-ubyte"

try:
    with open(file_path, 'rb') as file:
        data = file.read(16)
        print("File opened successfully. First 16 bytes:", data)
except Exception as e:
    print("Error opening file:", e)
