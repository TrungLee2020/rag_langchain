import os
from dotenv import load_dotenv

load_dotenv()
hf_key = os.getenv('HUGF_KEY')
def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value
