from typing import (
    List,
    Tuple,
    Callable
)
from pydantic import BaseModel
import datetime
import re
from dateutil.parser import parse as parse_date
from nltk.tokenize import word_tokenize
from nltk.stem import RegexpStemmer
import pandas as pd
import numpy as np


class ParsingException(Exception):
    pass


class Message(BaseModel):
    time: datetime.datetime
    author: str
    content: str

    # usage
    def tokenise(self, stem: bool = False) -> List[str]:
        words = word_tokenize(self.content)
        if stem:
            stemmer = RegexpStemmer(
                'ing$|s$|ed$|y$|er$|[^aeiou]{1}y$|e$', min=3
            )
            words = [stemmer.stem(word) for word in words]
        return words

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.tokenise())
    
    def __lt__(self, other):
        return self.time < other
    
    def __gt__(self, other):
        return self.time > other
    
    def __le__(self, other):
        return self.time <= other
    
    def __ge__(self, other):
        return self.time >= other
    
    def __str__(self):
        return f'{self.time} - {self.author}: {self.content}'

    @classmethod
    def from_line(cls, line: str):
        try:
            time, author, content = cls.parse_line(line)
        except:
            raise ParsingException(f'Could not parse line: {bytes(line, encoding="utf-8")} | Length: {len(line)}')
        return cls(
            time=time,
            author=author,
            content=content
        )
    
    @staticmethod
    def parse_line(line: str) -> Tuple[datetime.datetime, str, str]:
        p_reg = re.compile(r'(^[0-9]{2}/[0-9]{2}/[0-9]{4}\, [0-9]{2}\:[0-9]{2}) - (.*?)\: (.*)$')
        try:
            time, author, content = p_reg.search(line).groups()
        except: 
            # likely to be a system message
            sys_reg = re.compile(r'(^[0-9]{2}/[0-9]{2}/[0-9]{4}\, [0-9]{2}\:[0-9]{2}).*((?<=\- ).*$)')
            time, content = sys_reg.search(line).groups()
            author = '_SYSTEM_'
        
        # clean up a bit with right date type and remove newlines
        time = parse_date(time, dayfirst=True)
        content = content.rstrip()

        return time, author, content
        
        

class Chat(object):

    def __init__(self, lines: List[str] = None):
        if lines:
            self._raw_lines = lines
            self.messages = self._build_messages(lines)
        self.create_df()


    @classmethod
    def from_file(cls, message_file: str):
        with open(message_file, 'r', encoding="utf8") as f:
            lines = f.readlines()
        return cls(lines)

    
    # usage
    def create_df(self) -> pd.DataFrame:
        """
        Convert messages to pandas dataframe
        """
        data = [
            {
                'time':m.time, 
                'author': m.author, 
                'content': m.content, 
                'tokens': m.tokenise()
            }
            for m in self.messages
        ]
        self.df = pd.DataFrame(data)

    def get_messages_by_author(self, author: str) -> List[Message]:
        return [m for m in self.messages if m.author == author]
    
    def get_tokens_by_author(
            self, 
            author: str, 
            stem: bool = False) -> List[List[str]]:
        """
        Return a list of lists with word tokens per message
        """
        return [m.tokenise(stem=stem) for m in self.messages if m.author == author]

    # built-in analytical methods
    def get_authors_by_words(self, words: List[str]) -> dict:
        """
        return a dict with the counts of times a word in the supplied
        word list has been used
        """

        authors = {name: 0 for name in self.participants}
        for message in self.messages:
            no_mentioned_words = len([w for w in message.tokenise(True) if w in words])
            authors[message.author]+=no_mentioned_words
        return authors

    def get_contributions_by_author(self) -> dict:
        """
        return a dict counter of contributions per author
        """
        authors = {name: 0 for name in self.participants}
        for message in self.messages:
            authors[message.author]+=1
            
        return authors

    # methods sepcifically to add new data to the df
    def get_authors_by_verbosity(
            self, aggfunc: Callable = sum, add_to_df: bool = True) -> dict:
        """
        returns a dict of authors and the return of an aggregate function
        applied to the lengths of content of the message
        """
        authors = {name: [] for name in self.participants}
        if add_to_df:
            if self.df.empty:
                self.create_df()

        for i, m in enumerate(self.messages):
            # get content length of the message
            content_langth = len(m.content)
            authors[m.author].append(content_langth)
            if add_to_df:
                self.df.loc[i, 'content_length'] = content_langth
        
        # aggregate as per func supplied
        return {k:aggfunc(v) for k,v in authors.items()}

    
    # non-usage
    def _build_messages(self,lines: List[str]):
        """
        needed to check that messages broken across several lines
        are considered part of the same message
        """
        message_lines = []
        for line in lines:
            if self._is_new_message(line):
                message_lines.append(line.rstrip())
            else:
                message_lines[-1]+=' '+line.rstrip()
        return message_lines
    
    @staticmethod
    def _is_new_message(line: str) -> bool:
        # starts with a date?
        dt_reg = r'^[0-9]{2}/[0-9]{2}/[0-9]{4}'
        r = re.compile(dt_reg)
        return bool(r.match(line))

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, lines: List[str]):
        self._messages = [ Message.from_line(line) for line in lines if line != '\n']
        self.participants = set(m.author for m in self._messages)
    
    # dunders
    def __len__(self):
        return len(self.messages)
    
    def __repr__(self) -> str:
        return str(self.create_df())
    
    # support subscripts - return a new chat for a smaller slice
    def __getitem__(self, key):
        if isinstance(key,slice):
            lines = self._raw_lines[key]
            return Chat(lines)
        else:
            return self.messages[key]
