from torchvision import transforms, datasets

# Load the dataset
test_data = datasets.FashionMNIST(root='./data', train=False, download=True)

# Get one image and its label
img, label = test_data[0]

# Display the image and label
img.show()  # Opens the image using the default image viewer
img.save("sample_fashion_item.png")  # Save the image locally

print(f"Label: {label}")