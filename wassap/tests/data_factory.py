from faker import Faker
from typing import List
import datetime
import random

f = Faker()

def make_date():
    return datetime.datetime(
        2021,
        random.randint(1, 12),
        random.randint(1, 28),
        random.randint(0, 23),
        random.randint(0, 59)
        ).strftime('%d/%m/%Y, %H:%M') 

def create_message_lines(n: int) -> List[str]:
    return [(
        f'{make_date()} - {f.first_name()}: {f.sentence()}'
    ) for _ in range(n)]
