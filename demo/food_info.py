from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import requests
import json
import os

# loading env
load_dotenv()
API_KEY = os.getenv('API_KEY')

# Matching classes with FDC ID
food_list_id = {'chicken curry': '2341861', 
                'chocolate cake': '2343346', 
                'hamburger': '2342374', 
                'pizza': '2344102', 
                'ramen': '2341959'}

# Function to provide info on the food
def fds_food_info(food_name: str):
    """
    This function helps in collecting food info from the FDC database.
    Parameters:
        food_name: A string containing one of the classes.
    Returns: 
        food_info: A string containing all the nutritional info.
    """
    url = 'https://api.nal.usda.gov/fdc/v1/food'
    fdc_id = food_list_id[food_name]

    response = requests.get(url + '/' + fdc_id, 
                            headers={'Accept': 'application/json'}, 
                            auth=HTTPBasicAuth(API_KEY, ''))
    
    obj = json.loads(response.text) if response.status_code == 200 else None

    if obj != None:
        info_list = []
        for i in obj['foodNutrients']:
            info_list.append(f"{i['nutrient']['name']}: {i['amount']}")
        
        food_info = f'Food Name: {food_name}\nNutritional Info:\n\n' + ',\n'.join([str(elem) for i, elem in enumerate(info_list)])
        return food_info
    else:
        return response, obj, "There's no information for this food."

