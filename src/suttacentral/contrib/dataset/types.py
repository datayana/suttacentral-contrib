from dataclasses import dataclass


@dataclass
class SCDatasetPath:
    directory: str
    lang: str
    corpus: str
    encoding: str = "utf-8"


@dataclass
class SCDataFilePath:
    file_path: str
    lang: str
    corpus: str
    encoding: str = "utf-8"


@dataclass
class SCTextData:
    id: str
    corpus_id: str
    text_id: str
    paragraph_id: str
    pali: str = None
    english: str = None
