from enum import Enum

class TaskKey(str, Enum):
    SPAM_DETECTION = "spam_detection"
    HAM_INTENT = "ham_intent"