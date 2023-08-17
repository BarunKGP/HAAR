DATA_ROOT = "../2g1n6qdydwa9u22shpxqzp0t8m/"
VERB_CLASSES = "../epic-kitchens-100-annotations/EPIC_100_verb_classes.csv"
NOUN_CLASSES = "../epic-kitchens-100-annotations/EPIC_100_noun_classes_v2.csv"

PICKLE_ROOT = "../pickles"  # location of pickles directory on server
TRAIN_PICKLE = "data/train100_version_fix.pkl"  # location of pickle file to generate the training dataset
VALIDATION_PICKLE = ""  # location of pickle file to generate the validation dataset
TEST_PICKLE = "data/test100_version_fix.pkl"  # location of pickle file to generate the test dataset

STRIDE = 3
WORD_EMBEDDING_SIZE = 384
MULTIMODAL_FEATURE_SIZE = 2430  # number of concatenated features in each feature vector
NUM_VERBS = 97
NUM_NOUNS = 305  # nouns_classes_v2.csv
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

TRAIN_LOGNAME = "data/pilot-02/logs/train_temp_transformer.log"
RECOG_LOGNAME = "data/pilot-02/logs/recog_temp_transformer.log"

DEFAULT_OPT="Adam"
DEFAULT_ARGS={
    "weight_decay": 5e-4
}
D_MODEL=512
