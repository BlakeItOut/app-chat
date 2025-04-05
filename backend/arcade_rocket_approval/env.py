import os

from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ARCADE_API_KEY = os.getenv("ARCADE_API_KEY")
ARCADE_BASE_URL = os.getenv("ARCADE_BASE_URL")


# Rocket Mortgage API
APPROVAL_BASE_URL = "https://application.rocketmortgage.com"
