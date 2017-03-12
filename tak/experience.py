import MySQLdb
import os
import peewee
from peewee import Model, MySQLDatabase

db = MySQLDatabase(
    os.getenv('TAK_DB_SCHEMA', 'tak'),
    host=os.getenv('TAK_DB_HOST', 'localhost'),
    user=os.getenv('TAK_DB_USER', 'root'),
    passwd=os.getenv('TAK_DB_PASSWD', '')
)

class Experience(Model):

    state = peewee.TextField()
    action = peewee.TextField()
    reward = peewee.IntegerField()
    state_prime = peewee.TextField()
    player = peewee.IntegerField()
    player_prime = peewee.IntegerField()
    env_id = peewee.CharField()

    class Meta:
        database = db

db.connect()
# Experience.create_table()