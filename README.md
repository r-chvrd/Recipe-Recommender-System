# Recipe Recommender System
This is a recipe recommender built off of Jamie Oliver's recipes on https://www.jamieoliver.com/ 
This application allows the user to enter n amount of ingredients, their macronutritional and caloric minimums, which will then output a dataframe that meets their requirements.


## Libraries required (Python)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- re
- nltk
- wordcloud
- datetime
- unidecode
- string
- ast
- bs4
- time
- requests
- streamlit



## Streamlit

Run the .py file in streamlit using VSCode terminal with the libraries above installed.
To run the streamlit app, use:

streamlit run modelling2.py


## Jupyter Notebook
For jupyter notebook followthrough, run the following notebooks in this order:

1) WebscrapingJO.ipynb
(Keep in mind, this process will take 40-50 minutes depending on your system, so i suggest installing the Jamie_Oliver_Recipes.csv and move onto step 2, or install Cleaned_data.csv and skip to step 3).
   
2) NLP_DC.ipynb

3) Modelling.ipynb

## Inspirations

This project was inspired by:
- https://towardsdatascience.com/building-a-recipe-recommendation-system-297c229dda7b 


## License
[GNU General Public License version 3](https://opensource.org/license/gpl-3-0/)

