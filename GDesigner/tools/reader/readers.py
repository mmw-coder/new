import json
from pathlib import Path
from abc import ABC, abstractmethod
from GDesigner.utils.log import logger

class Reader(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """ To be overriden by the descendant class """

class JSONLReader(Reader):
    @staticmethod
    def parse_file(file_path: Path) -> list:
        logger.info(f"Reading JSON Lines file from {file_path}.")
        with open(file_path, "r",encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
        return lines
    
    def parse(file_path: Path) -> str:
        logger.info(f"Reading JSON Lines file from {file_path}.")
        with open(file_path, "r",encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            text = '\n'.join([str(line) for line in lines])
        return text
