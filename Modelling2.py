# Import necessary libraries
import streamlit as st
import pandas as pd
import re
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import string
from sklearn.metrics.pairwise import cosine_similarity


pd.set_option('display.max_colwidth', None)
df = pd.read_csv('Cleaned_data.csv', index_col=False)
del df['Unnamed: 0'] # Delete 'Unnamed: 0' column



# Set up TfidfVectorizer and other preprocessing steps
tfidf = TfidfVectorizer()
tfidf.fit(df['Ingredients_final'].values.astype('U'))
tfidf_recipe = tfidf.transform(df['Ingredients_final'])

# Define function that parses user input ingredients
def ingredient_parser(ingredients):
    measures = ['teaspoon', 't', 'tsp.', 'tsp', 'tablespoon', 'T', 'tbl.', 'tb', 'tbsp.','tbsp', 'fluid ounce', 'fl oz', 'gill', 'cup', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'ml', 'milliliter', 'millilitre', 'cc', 'mL', 'l', 'liter', 'litre', 'L', 'dl', 'deciliter', 'decilitre', 'dL', 'bulb', 'level', 'heaped', 'rounded', 'whole', 'pinch', 'medium', 'slice', 'pound', 'lb', '#', 'ounce', 'oz', 'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram', 'kilogramme', 'x', 'of', 'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre', 'm', 'meter', 'metre', 'inch', 'in', 'milli', 'centi', 'deci', 'hecto', 'kilo']
    bad_words = ['oil', 'fresh', 'olive', 'a', 'red', 'sauce', 'clove', 'or', 'pepper', 'bunch', 'salt', 'and', 'leaf', 'chilli', 'large', 'extra', 'water', 'white', 'ground', 'dried', 'sprig', 'small', 'free', 'handful', 'sugar', 'from', 'virgin', 'soy', 'black', 'chopped', 'vinegar', 'green', 'piece', 'seed', 'for', 'sustainable', 'range', 'cornstarch', 'higher', 'welfare', 'to', 'peeled', 'sesame', 'flour', 'tin', 'finely', 'the', 'freshly', 'bean', 'quality', 'few', 'ripe', 'parsley', 'sea', 'stock', 'source', 'flat', 'vegetable', 'smoked', 'organic', 'spring', 'fillet', 'sliced', 'plus', 'corn', 'plain', 'stick', 'cooking', 'light', 'picked', 'mixed', 'your', 'powder', 'bay', 'optional', 'baby', 'serve', 'stalk', 'unsalted', 'plum', 'natural', 'fat', 'fish', 'sweet', 'skin', 'such', 'juice', 'ask', 'brown', 'into', 'on', 'yellow', 'roughly', 'cut', 'good', 'dark', 'chili', 'orange', 'cherry', 'grated', 'frozen', 'bread', 'trimmed', 'breast', 'low', 'knob', 'dusting', 'salad', 'bell', 'cooked', 'runny', 'deseeded', 'balsamic', 'with', 'paste', 'bouillon', 'curry', 'streaky', 'use', 'pin', 'rasher', 'nut', 'cream', 'if', 'groundnut', 'soft', 'you', 'squash', 'tamari', 'chinese', 'zest', 'baking', 'grating', 'bone', 'hot', 'steak', 'boiling', 'minced', 'thigh', 'can', 'other', 'colour', 'shiitake', 'puree', 'dry', 'halved', 'skinless', 'spice', 'amount', 'chive', 'tinned', 'english', 'butternut', 'splash', 'shoulder', 'king', 'leftover', 'washed', 'firm', 'thick', 'flake', 'stir', 'broth', 'caper', 'big', 'dijon', 'is', 'little', 'pastry', 'five', 'sized', 'fishmonger', 'deep', 'removed', 'any', 'cube', 'frying', 'raw']

    translator = str.maketrans('', '', string.punctuation) # Get rid of punctuations using maketrans
    lemmatizer = WordNetLemmatizer() # Set lemmatizer
    new_ingred_list = [] # Empty list for parsed ingredients

    # Loop through each ingredient
    for i in ingredients:
        i.translate(translator) # Get rid of punctuation
        items = re.split(' |-', i) # Split with where there is a space or hyphenated words
        items = [word for word in items if word.isalpha()] # Get rid of non alphabet words
        items = [word.lower() for word in items] # Make everything lower case
        items = [unidecode.unidecode(word) for word in items] # Unidecode each word
        items = [lemmatizer.lemmatize(word) for word in items] # Lemmatize each word
        items = [word for word in items if word not in measures] # Take out measure words
        items = [word for word in items if word not in bad_words] # Take out bad words

        new_ingred_list.append(' '.join(items))# Append to list

    return ' '.join(new_ingred_list) # Return new parsed ingredients

# Define a function that gets cosine similarity scores with each recipe, taking in ingredient parameters
# Define a function that gets cosine similarity scores with each recipe, taking in ingredient parameters
def get_scores(i):
    input_parsed = ingredient_parser(i)
    # Use pretrained tfidf model to encode our input ingredients
    ingredients_tfidf = tfidf.transform([input_parsed])
    # Calculate cosine similarity between actual recipe ingredients and test ingredients
    cos_sim = cosine_similarity(tfidf_recipe, ingredients_tfidf)
    return cos_sim


# Define a function to get user input for caloric limit, minimum carbs, fats, and protein
def get_user_nutrient_limits():
    calorie_limit = st.sidebar.number_input('Enter the caloric limit for the recipes (or enter 0 for no limit): ', min_value=0)
    min_protein = st.sidebar.number_input('Enter your minimum protein requirement (in grams): ', min_value=0)
    min_fats = st.sidebar.number_input('Enter your minimum fats requirement (in grams): ', min_value=0)
    min_carbs = st.sidebar.number_input('Enter your minimum carbs requirement (in grams): ', min_value=0)

    return calorie_limit, min_protein, min_fats, min_carbs

# Define a function to get recommendations based on user input
def get_recommendations(scores, N=5, calorie_limit=None, min_protein=None, min_fats=None, min_carbs=None):
    df_copy = df.copy()  # Copy the dataframe
    df_copy['Rec_score'] = scores  # Put scores into the dataframe

    # Print the available columns for debugging
    print("Available columns:", df_copy.columns)

    # Filter by caloric limit if provided
    if calorie_limit is not None and 'Calories' in df_copy.columns:
        df_copy = df_copy[df_copy['Calories'].astype(float) <= calorie_limit]

    # Filter by minimum protein if provided
    if min_protein is not None and 'Protein/g' in df_copy.columns:
        df_copy = df_copy[df_copy['Protein/g'].astype(float) >= min_protein]

    # Filter by minimum fats if provided
    if min_fats is not None and 'Fats/g' in df_copy.columns:
        df_copy = df_copy[df_copy['Fats/g'].astype(float) >= min_fats]

    # Filter by minimum carbs if provided
    if min_carbs is not None and 'Carbs/g' in df_copy.columns:
        df_copy = df_copy[df_copy['Carbs/g'].astype(float) >= min_carbs]

    # Order the scores with 'Rec_score' and sort to get the highest N scores
    top = df_copy.sort_values('Rec_score', ascending=False).head(N)

    # Print the available columns for debugging
    print("Selected columns:", top.columns)

    # Create a dataframe to load in recommendations
    recommendation = top[['Title', 'Ingredients', 'Servings', 'Difficulty', 'Calories', 'Fats/g', 'Protein/g', 'Carbs/g', 'Time', 'URL', 'Rec_score']]
    return recommendation

# Streamlit app
def main():
    st.title('Recipe Recommender')
    st.subheader('By Richard V')

    # Sidebar
    st.sidebar.title('User Input')
    
    user_ingreds = []
    user_input = st.sidebar.text_input('Enter an ingredient (type "done" when finished):')
    
    while user_input and user_input.lower() != 'done':
        user_ingreds.append(user_input.lower())
        user_input = st.sidebar.text_input(f'Enter ingredient #{len(user_ingreds) + 1} (type "done" when finished):', key=len(user_ingreds))

    # Get user input for recommendations
    N = st.sidebar.slider('Number of recipes to recommend', 1, 10, 5)
    calorie_limit, min_protein, min_fats, min_carbs = get_user_nutrient_limits()

    # Calculate scores and recommendations
    if user_ingreds:
        scores = get_scores(user_ingreds)
        recommendations = get_recommendations(scores, N, calorie_limit, min_protein, min_fats, min_carbs)

        # Display recommendations
        st.subheader('Recommendations:')
        st.table(recommendations[['Title', 'Servings', 'Difficulty', 'Calories', 'Fats/g', 'Protein/g', 'Carbs/g', 'Time', 'URL', 'Rec_score']])
    else:
        st.warning('Please enter at least one ingredient.')
        
    
    print(df.columns)
if __name__ == '__main__':
    main()

