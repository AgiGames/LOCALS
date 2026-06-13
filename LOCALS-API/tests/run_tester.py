from locals import run

run(seed=1111, train_split=0.8, test_split=0.1, images_dir='dataset/images', labels_dir='dataset/labels', num_epochs=2)

print("\033[92mSUCCESS\033[0m")