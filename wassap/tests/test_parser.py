
import datetime
from .data_factory import create_message_lines
import pytest
from ..parser import Message, Chat

import numpy as np

# parsing a single line into 3 objects: date, author and message
def test_parse_line():
    l = '27/04/2021, 08:18 - Kyle: Number gas poor nothing will statement.'
    # action
    message_date, message_author, message_content = Message.parse_line(l, day_first=True)

    # assert
    assert isinstance(message_date,datetime.datetime)
    assert len(message_author)>0
    assert len(message_content)>0

def test_multiple_parse_line():
    # arrange
    lines = create_message_lines(100) # 100 lines

    # act
    for l in lines:
        time, author, content = Message.parse_line(l, day_first=True)

        # assert
        assert isinstance(time,datetime.datetime)
        assert len(author)>0
        assert len(content)>0

def test_is_new_message_happy():
    # arr
    text = r'28/08/2020, 18:33 - John: Maybe going through them might be ok'

    res = Chat._is_new_message(text)

    assert res

def test_is_new_message_happy_stringified_message():
    # arr
    text = r'2020-08-28 18:33 - John: Maybe going through them might be ok'

    res = Chat._is_new_message(text)

    assert res


def test_is_new_message_unhappy():
    # arr
    text = r'date missing here - John: Maybe going through them might be ok'

    res = Chat._is_new_message(text)

    assert not res


def test_build_messages_no_line_breaks():
    lines = [
        '28/08/2020, 18:33 - John: line 1',
        '28/08/2020, 18:33 - John: line 2',
        '28/08/2020, 18:33 - John: line 3',
    ]
    c = Chat(lines)

    assert len(c) == 3
    for i, m in enumerate(c.messages):
        assert m.content == f'line {i+1}'

def test_build_messages_with_line_breaks():
    lines = [
        '28/08/2020, 18:33 - John: line 1',
        'continuation of line 1',
        '28/08/2020, 18:33 - John: line 2',
    ]
    c = Chat(lines)

    assert len(c) == 2
    assert c.messages[0].content == 'line 1 continuation of line 1'
    assert c.messages[1].content == 'line 2'


def test_tokenise():
    # arrange
    m = Message(
        time=datetime.datetime.now(),
        author='test',
        content='words to tokenise'
    )
    # act
    tokens = m.tokenise()

    # assert
    assert tokens == ['words','to','tokenise']

def test_tokenise_using_stemmer():
    # arrange
    m = Message(
        time=datetime.datetime.now(),
        author='test',
        content='words to tokenise'
    )
    # act
    tokens = m.tokenise(stem=True)

    # assert
    assert tokens == ['word','to','tokenis']

def test_to_df():
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: line 1',
        '28/08/2020, 18:33 - John: line 2',
        '28/08/2020, 18:33 - John: line 3',

    ]
    c = Chat(lines)

    # act
    df = c.df

    # assert
    for t in df.time:
        assert t == datetime.datetime(2020,8,28,18,33)
    for a in df.author:
        assert a == 'John'
    for i, c in enumerate(df.content):
        assert c == f'line {i+1}'



@pytest.fixture
def word_list():
    return ["beach", "sea", "ice cream", "bat"]

def test_list_by_words_expected(word_list):
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: beach sea',
        '28/08/2020, 18:34 - Jack: bat',
        '28/08/2020, 18:35 - John: we beaching today?',

    ]
    c = Chat(lines)

    # act
    word_counter = c.get_authors_by_words(word_list)

    # assert 
    assert word_counter['Jack'] == 1
    assert word_counter['John'] == 3 

def test_list_by_words_one_not_used(word_list):
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: sea bat',
        '28/08/2020, 18:34 - Jack: Im well spoken',
        '28/08/2020, 18:35 - John: we beaching today?',

    ]
    c = Chat(lines)

    # act
    word_counter = c.get_authors_by_words(word_list)

    # assert 
    assert word_counter['Jack'] == 0
    assert word_counter['John'] == 3 

def test_list_by_words_none_present(word_list):
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: hello jack',
        '28/08/2020, 18:34 - Jack: Im well spoken',
        '28/08/2020, 18:35 - John: right you are!',

    ]
    c = Chat(lines)

    # act
    word_counter = c.get_authors_by_words(word_list)

    # assert 
    assert word_counter['Jack'] == 0
    assert word_counter['John'] == 0
         
def test_list_by_words_stem_variation(word_list):
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: beacher beaching beached',
        '28/08/2020, 18:34 - Jack: sea seas',

    ]
    c = Chat(lines)

    # act
    word_counter = c.get_authors_by_words(word_list)

    # assert 
    assert word_counter['John'] == 3
    assert word_counter['Jack'] == 2

def test_get_contributors():
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: hello jack old pal',
        '28/08/2020, 18:34 - Jack: sup mate',
        '28/08/2020, 18:35 - Gerald: what about me',
        '28/08/2020, 18:35 - John: what about you?',
        '28/08/2020, 18:35 - John: lols',
        '28/08/2020, 18:36 - Jack: looool',
    ]
    c = Chat(lines)

    # act
    counter = c.get_contributions_by_author()

    # assert 
    assert counter['John'] == 3
    assert counter['Jack'] == 2
    assert counter['Gerald'] == 1

def test_get_author_total_content_length():
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: hello jack old pal',
        '28/08/2020, 18:34 - Jack: sup mate',
        '28/08/2020, 18:35 - Gerald: what about me',
        '28/08/2020, 18:35 - John: what about you?',
        '28/08/2020, 18:35 - John: lols',
        '28/08/2020, 18:36 - Jack: looool',
    ]

    # act
    c = Chat(lines)
    counter = c.get_authors_by_verbosity()

    #assert 
    assert counter['John'] == 37
    assert counter['Jack'] == 14
    assert counter['Gerald'] == 13

def test_get_author_average_content_length():
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: hello jack old pal',
        '28/08/2020, 18:34 - Jack: sup mate',
        '28/08/2020, 18:35 - Gerald: what about me',
        '28/08/2020, 18:35 - John: what about you?',
        '28/08/2020, 18:35 - John: lols',
        '28/08/2020, 18:36 - Jack: looool',
    ]

    # act
    c = Chat(lines)
    counter = c.get_authors_by_verbosity(aggfunc = np.mean)

    #assert 
    assert int(counter['John']) == 12
    assert int(counter['Jack']) == 7
    assert int(counter['Gerald']) == 13

def test_subscript_multiple_lines():
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: hello jack old pal',
        '28/08/2020, 18:34 - Jack: sup mate',
        '28/08/2020, 18:35 - Gerald: what about me',
        '28/08/2020, 18:35 - John: what about you?',
        '28/08/2020, 18:35 - John: lols',
        '28/08/2020, 18:36 - Jack: looool',
    ]

    # act
    c = Chat(lines)

    subbed = c[:2]

    #assert 
    assert isinstance(subbed, Chat)
    assert len(subbed) == 2
    assert subbed.participants == {'John', 'Jack'}

def test_subscript_single_value():
    # arrange
    lines = [
        '28/08/2020, 18:33 - John: hello jack old pal',
        '28/08/2020, 18:34 - Jack: sup mate',
        '28/08/2020, 18:35 - Gerald: what about me',
        '28/08/2020, 18:35 - John: what about you?',
        '28/08/2020, 18:35 - John: lols',
        '28/08/2020, 18:36 - Jack: looool',
    ]

    # act
    c = Chat(lines)

    subbed = c[1]

    #assert 
    assert isinstance(subbed, Message)
    assert subbed.author == 'Jack'
    assert subbed.content == 'sup mate'
