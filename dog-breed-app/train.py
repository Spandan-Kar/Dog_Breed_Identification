from classifier import DogBreedClassifier

clf = DogBreedClassifier(
    labels_csv_path="../dog-breed-identification/labels.csv",
    images_dir="../dog-breed-identification/train/"
)

clf.create_model()
clf.train(epochs=25)