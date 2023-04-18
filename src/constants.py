DATA_ROOT = "../2g1n6qdydwa9u22shpxqzp0t8m/"
PICKLE_ROOT = "../pickles"  # location of pickles directory on server
STRIDE = 3
BATCH_SIZE = 8
WORD_EMBEDDING_SIZE = 384
MULTIMODAL_FEATURE_SIZE = 4094  # number of concatenated features in each feature vector
NUM_VERBS = 97
NUM_NOUNS = 305  # nouns_classes_v2.csv
VERB_CLASSES = "../epic-kitchens-100-annotations/EPIC_100_verb_classes.csv"
NOUN_CLASSES = "../epic-kitchens-100-annotations/EPIC_100_noun_classes_v2.csv"

TRAIN_PICKLE = "data/train100_version_fix.pkl"  # location of pickle file to generate the training dataset
VALIDATION_PICKLE = ""  # location of pickle file to generate the validation dataset
TEST_PICKLE = "data/test100_version_fix.pkl"  # location of pickle file to generate the test dataset

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
