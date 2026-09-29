import pymongo
from dotenv import load_dotenv
import os


load_dotenv()

myclient = pymongo.MongoClient(os.getenv("MONGODB_URI"))

mydb = myclient["cin_mind"]

mycol = mydb['cleaned_data']

