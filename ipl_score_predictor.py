# import math
# import numpy as np
# import pickle
# import streamlit as st

# # Set Page Wide
# st.set_page_config(page_title='IPL_Score_Predictor', layout='centered')

# # Get the ML Model
# filename = 'ml_model.pkl'
# model = pickle.load(open(filename, 'rb'))

# # Title of the page with CSS
# st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor</h1>", unsafe_allow_html=True)

# # Add background image
# st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# #Add description

# with st.expander("Description"):
#     st.info("""A Simple ML Model to predict IPL Scores between teams in an ongoing match. To make sure the model results accurate score and some reliability the minimum no. of current overs considered is greater than 5 overs.
    
#  """)

# # SELECT THE BATTING TEAM


# batting_team= st.selectbox('Select the Batting Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))

# prediction_array = []
#   # Batting Team
# if batting_team == 'Chennai Super Kings':
#     prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
# elif batting_team == 'Delhi Daredevils':
#     prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
# elif batting_team == 'Kings XI Punjab':
#     prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
# elif batting_team == 'Kolkata Knight Riders':
#     prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
# elif batting_team == 'Mumbai Indians':
#     prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
# elif batting_team == 'Rajasthan Royals':
#     prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
# elif batting_team == 'Royal Challengers Bangalore':
#     prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
# elif batting_team == 'Sunrisers Hyderabad':
#     prediction_array = prediction_array + [0,0,0,0,0,0,0,1]




# #SELECT BOWLING TEAM

# bowling_team = st.selectbox('Select the Bowling Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))
# if bowling_team==batting_team:
#     st.error('Bowling and Batting teams should be different')
# # Bowling Team
# if bowling_team == 'Chennai Super Kings':
#     prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
# elif bowling_team == 'Delhi Daredevils':
#     prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
# elif bowling_team == 'Kings XI Punjab':
#     prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
# elif bowling_team == 'Kolkata Knight Riders':
#     prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
# elif bowling_team == 'Mumbai Indians':
#     prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
# elif bowling_team == 'Rajasthan Royals':
#     prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
# elif bowling_team == 'Royal Challengers Bangalore':
#     prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
# elif bowling_team == 'Sunrisers Hyderabad':
#     prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
  

# col1, col2 = st.columns(2)

# #Enter the Current Ongoing Over
# with col1:
#     overs = st.number_input('Enter the Current Over',min_value=5.1,max_value=19.5,value=5.1,step=0.1)
#     if overs-math.floor(overs)>0.5:
#         st.error('Please enter valid over input as one over only contains 6 balls')
# with col2:
# #Enter Current Run
#     runs = st.number_input('Enter Current runs',min_value=0,max_value=354,step=1,format='%i')


# #Wickets Taken till now
# wickets =st.slider('Enter Wickets fallen till now',0,9)
# wickets=int(wickets)

# col3, col4 = st.columns(2)

# with col3:
# #Runs in last 5 over
#     runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs',min_value=0,max_value=runs,step=1,format='%i')

# with col4:
# #Wickets in last 5 over
#     wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs',min_value=0,max_value=wickets,step=1,format='%i')

# #Get all the data for predicting

# prediction_array = prediction_array + [runs, wickets, overs, runs_in_prev_5,wickets_in_prev_5]
# prediction_array = np.array([prediction_array])
# predict = model.predict(prediction_array)


# if st.button('Predict Score'):
#     #Call the ML Model
#     my_prediction = int(round(predict[0]))
    
#     #Display the predicted Score Range
#     # x=f'PREDICTED MATCH SCORE : {my_prediction-5} to {my_prediction+5}' 
#     x = f'PREDICTED MATCH SCORE : {my_prediction}'
#     st.success(x)


import math
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the pitch conditions dataset
file_path = "pitch_conditions.csv"
pitch_conditions = pd.read_csv(file_path)

# Set Page Wide
st.set_page_config(page_title='IPL_Score_Predictor', layout='centered')

# Load the ML Model
filename = 'ml_model.pkl'
model = pickle.load(open(filename, 'rb'))

# Title of the page with CSS
st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor</h1>", unsafe_allow_html=True)

# Add background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add description
with st.expander("Description"):
    st.info("""A Simple ML Model to predict IPL Scores between teams in an ongoing match. To ensure accuracy and reliability, the minimum overs considered is greater than 5 overs.""")

# Venue Selection
venue = st.selectbox("Select the Venue", pitch_conditions["venue"].unique())

# Display the pitch description
venue_info = pitch_conditions[pitch_conditions["venue"] == venue]
if not venue_info.empty:
    description = venue_info["description"].values[0]
    st.write(f"**Pitch Description:** {description}")
else:
    st.write("No pitch data available for this venue.")

# Team Selection
batting_team = st.selectbox('Select the Batting Team', (
    'Chennai Super Kings', 'Delhi Capitals', 'Punjab Kings', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
))

bowling_team = st.selectbox('Select the Bowling Team', (
    'Chennai Super Kings', 'Delhi Capitals', 'Punjab Kings', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
))

if batting_team == bowling_team:
    st.error('Bowling and Batting teams should be different')

# Feature Input
col1, col2 = st.columns(2)
with col1:
    overs = st.number_input('Enter the Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    if overs - math.floor(overs) > 0.5:
        st.error('Please enter a valid over input (one over contains 6 balls)')

with col2:
    runs = st.number_input('Enter Current Runs', min_value=0, max_value=354, step=1, format='%i')

wickets = st.slider('Enter Wickets fallen till now', 0, 9)
wickets = int(wickets)

col3, col4 = st.columns(2)
with col3:
    runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs', min_value=0, max_value=runs, step=1, format='%i')

with col4:
    wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs', min_value=0, max_value=wickets, step=1, format='%i')

# Encoding the Teams
team_encoding = {
    'Chennai Super Kings': [1, 0, 0, 0, 0, 0, 0, 0],
    'Delhi Capitals': [0, 1, 0, 0, 0, 0, 0, 0],
    'Punjab Kings': [0, 0, 1, 0, 0, 0, 0, 0],
    'Kolkata Knight Riders': [0, 0, 0, 1, 0, 0, 0, 0],
    'Mumbai Indians': [0, 0, 0, 0, 1, 0, 0, 0],
    'Rajasthan Royals': [0, 0, 0, 0, 0, 1, 0, 0],
    'Royal Challengers Bangalore': [0, 0, 0, 0, 0, 0, 1, 0],
    'Sunrisers Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1]
}

prediction_array = team_encoding[batting_team] + team_encoding[bowling_team]
prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]

# Convert to numpy array
prediction_array = np.array([prediction_array])

# Make Prediction
predict = model.predict(prediction_array)

if st.button('Predict Score'):
    my_prediction = int(round(predict[0]))
    st.success(f'PREDICTED MATCH SCORE: {my_prediction}')

# import math
# import numpy as np
# import pandas as pd
# import pickle
# import streamlit as st

# # Load the pitch conditions dataset
# file_path = "pitch_conditions.csv"
# pitch_conditions = pd.read_csv(file_path)

# # Set Page Wide
# st.set_page_config(page_title='IPL_Score_Predictor', layout='centered')

# # Load the ML Model
# filename = 'ml_model.pkl'
# model = pickle.load(open(filename, 'rb'))

# # Title of the page
# st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor</h1>", unsafe_allow_html=True)

# # Add background image
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
#         background-attachment: fixed;
#         background-size: cover;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Add description
# with st.expander("Description"):
#     st.info("""A Simple ML Model to predict IPL Scores between teams in an ongoing match. To ensure accuracy and reliability, the minimum overs considered is greater than 5 overs.""")

# # Venue Selection
# venue = st.selectbox("Select the Venue", pitch_conditions["venue"].unique())

# # Get pitch data
# venue_info = pitch_conditions[pitch_conditions["venue"] == venue]
# if not venue_info.empty:
#     description = venue_info["description"].values[0]
#     pitch_type = venue_info["pitch type"].values[0]
#     spin_friendly = venue_info["spin friendly"].values[0]
#     pace_friendly = venue_info["pace friendly"].values[0]

#     # Show pitch description
#     st.markdown(f"**Pitch Type:** {pitch_type}")
#     st.markdown(f"**Spin Friendly:** {'Yes' if spin_friendly == 'Yes' else 'No'}")
#     st.markdown(f"**Pace Friendly:** {'Yes' if pace_friendly == 'Yes' else 'No'}")
#     st.markdown(f"**Pitch Description:** {description}")
# else:
#     st.warning("No pitch data available for this venue.")
#     pitch_type = "Unknown"
#     spin_friendly = 0
#     pace_friendly = 0

# # Team Selection
# batting_team = st.selectbox('Select the Batting Team', (
#     'Chennai Super Kings', 'Delhi Capitals', 'Punjab Kings', 'Kolkata Knight Riders',
#     'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
# ))

# bowling_team = st.selectbox('Select the Bowling Team', (
#     'Chennai Super Kings', 'Delhi Capitals', 'Punjab Kings', 'Kolkata Knight Riders',
#     'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
# ))

# if batting_team == bowling_team:
#     st.error('Bowling and Batting teams should be different')

# # Feature Input
# col1, col2 = st.columns(2)
# with col1:
#     overs = st.number_input('Enter the Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
#     if overs - math.floor(overs) > 0.5:
#         st.error('Please enter a valid over input (one over contains 6 balls)')

# with col2:
#     runs = st.number_input('Enter Current Runs', min_value=0, max_value=354, step=1, format='%i')

# wickets = st.slider('Enter Wickets fallen till now', 0, 9)
# wickets = int(wickets)

# col3, col4 = st.columns(2)
# with col3:
#     runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs', min_value=0, max_value=runs, step=1, format='%i')

# with col4:
#     wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs', min_value=0, max_value=wickets, step=1, format='%i')

# # Encoding the Teams
# team_encoding = {
#     'Chennai Super Kings': [1, 0, 0, 0, 0, 0, 0, 0],
#     'Delhi Capitals': [0, 1, 0, 0, 0, 0, 0, 0],
#     'Punjab Kings': [0, 0, 1, 0, 0, 0, 0, 0],
#     'Kolkata Knight Riders': [0, 0, 0, 1, 0, 0, 0, 0],
#     'Mumbai Indians': [0, 0, 0, 0, 1, 0, 0, 0],
#     'Rajasthan Royals': [0, 0, 0, 0, 0, 1, 0, 0],
#     'Royal Challengers Bangalore': [0, 0, 0, 0, 0, 0, 1, 0],
#     'Sunrisers Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1]
# }

# # Prediction array
# prediction_array = team_encoding[batting_team] + team_encoding[bowling_team]
# prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]

# # Add pitch data (spin_friendly and pace_friendly as binary inputs)
# prediction_array += [spin_friendly, pace_friendly]

# # Convert to numpy array
# prediction_array = np.array([prediction_array])

# # Make Prediction
# predict = model.predict(prediction_array)

# if st.button('Predict Score'):
#     my_prediction = int(round(predict[0]))
#     st.success(f'PREDICTED MATCH SCORE: {my_prediction}')





# import math
# import numpy as np
# import pickle
# import pandas as pd
# import streamlit as st

# # Set Page Wide
# st.set_page_config(page_title='IPL Score Predictor', layout='centered')

# # Load the trained model
# filename = 'ml_model.pkl'
# model = pickle.load(open(filename, 'rb'))

# # Load the pitch conditions dataset
# pitch_conditions = pd.read_csv("pitch_conditions.csv")

# # Title of the page with CSS
# st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor</h1>", unsafe_allow_html=True)

# # Add background image
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
#         background-attachment: fixed;
#         background-size: cover;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Add description
# with st.expander("Description"):
#     st.info("""
#     A Simple ML Model to predict IPL Scores between teams in an ongoing match. 
#     To ensure accuracy, the minimum number of overs must be greater than 5.
#     """)

# # Venue selection dropdown
# venue = st.selectbox("Select Venue", pitch_conditions["venue"].unique())

# # Display pitch conditions
# if venue:
#     venue_info = pitch_conditions[pitch_conditions["venue"] == venue].iloc[0]
#     st.subheader(f"Pitch Conditions at {venue}:")
#     st.write(f"**Pitch Type:** {venue_info['pitch type']}")
#     st.write(f"**Avg 1st Innings Score:** {venue_info['avg first innings score']}")
#     st.write(f"**Spin Friendly:** {'Yes' if venue_info['spin friendly'] else 'No'}")
#     st.write(f"**Pace Friendly:** {'Yes' if venue_info['pace friendly'] else 'No'}")
#     st.write(f"**Description:** {venue_info['description']}")

# # Select Batting Team
# batting_team = st.selectbox('Select the Batting Team', (
#     'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
#     'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
#     'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
# ))

# # Encode Batting Team
# prediction_array = []
# team_mapping = {
#     'Chennai Super Kings': [1, 0, 0, 0, 0, 0, 0, 0],
#     'Delhi Daredevils': [0, 1, 0, 0, 0, 0, 0, 0],
#     'Kings XI Punjab': [0, 0, 1, 0, 0, 0, 0, 0],
#     'Kolkata Knight Riders': [0, 0, 0, 1, 0, 0, 0, 0],
#     'Mumbai Indians': [0, 0, 0, 0, 1, 0, 0, 0],
#     'Rajasthan Royals': [0, 0, 0, 0, 0, 1, 0, 0],
#     'Royal Challengers Bangalore': [0, 0, 0, 0, 0, 0, 1, 0],
#     'Sunrisers Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1]
# }

# prediction_array += team_mapping[batting_team]

# # Select Bowling Team
# bowling_team = st.selectbox('Select the Bowling Team', (
#     'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
#     'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
#     'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
# ))

# if bowling_team == batting_team:
#     st.error('Bowling and Batting teams should be different')

# # Encode Bowling Team
# prediction_array += team_mapping[bowling_team]

# col1, col2 = st.columns(2)

# # Enter Current Over
# with col1:
#     overs = st.number_input('Enter the Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
#     if overs - math.floor(overs) > 0.5:
#         st.error('Please enter a valid over input (one over contains 6 balls)')

# # Enter Current Runs
# with col2:
#     runs = st.number_input('Enter Current Runs', min_value=0, max_value=354, step=1, format='%i')

# # Wickets Taken
# wickets = st.slider('Enter Wickets Fallen', 0, 9)
# wickets = int(wickets)

# col3, col4 = st.columns(2)

# # Runs in last 5 overs
# with col3:
#     runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs', min_value=0, max_value=runs, step=1, format='%i')

# # Wickets in last 5 overs
# with col4:
#     wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs', min_value=0, max_value=wickets, step=1, format='%i')

# # Append other features to prediction array
# prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]

# # Append pitch conditions
# prediction_array += [
#     venue_info['pitch type'], 
#     venue_info['avg first innings score'], 
#     int(venue_info['spin friendly']), 
#     int(venue_info['pace friendly'])
# ]

# # Convert to numpy array for model prediction
# prediction_array = np.array([prediction_array])

# # Predict Score
# if st.button('Predict Score'):
#     predicted_score = int(round(model.predict(prediction_array)[0]))
    
#     # Display the predicted score
#     st.success(f'PREDICTED MATCH SCORE: {predicted_score}')

# import math
# import numpy as np
# import pickle
# import pandas as pd
# import streamlit as st

# # Set Page Wide
# st.set_page_config(page_title='IPL Score Predictor', layout='centered')

# # Load the trained model
# filename = 'ml_model.pkl'
# model = pickle.load(open(filename, 'rb'))


# # Load the pitch conditions dataset
# pitch_conditions = pd.read_csv("pitch_conditions.csv")

# # Title of the page with CSS
# st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor</h1>", unsafe_allow_html=True)

# # Add background image
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
#         background-attachment: fixed;
#         background-size: cover;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Add description
# with st.expander("Description"):
#     st.info("""
#     A Simple ML Model to predict IPL Scores between teams in an ongoing match. 
#     To ensure accuracy, the minimum number of overs must be greater than 5.
#     """)

# # Venue selection dropdown
# venue = st.selectbox("Select Venue", pitch_conditions["venue"].unique())

# # Get pitch conditions for the selected venue
# venue_info = pitch_conditions[pitch_conditions["venue"] == venue].iloc[0]

# # Pitch Conditions in an expandable dropdown
# with st.expander("Pitch Conditions"):
#     st.write(f"**Pitch Type:** {venue_info['pitch type']}")
#     st.write(f"**Avg 1st Innings Score:** {venue_info['avg first innings score']}")
#     st.write(f"**Spin Friendly:** {'Yes' if venue_info['spin friendly'] else 'No'}")
#     st.write(f"**Pace Friendly:** {'Yes' if venue_info['pace friendly'] else 'No'}")
#     st.write(f"**Description:** {venue_info['description']}")

# # Select Batting Team
# batting_team = st.selectbox('Select the Batting Team', (
#     'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
#     'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
#     'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Lucknow Super Giants', 'Gujrat Lions'
# ))

# # Encode Batting Team
# prediction_array = []
# team_mapping = {
#     'Chennai Super Kings': [1, 0, 0, 0, 0, 0, 0, 0,0,0],
#     'Delhi Daredevils': [0, 1, 0, 0, 0, 0, 0, 0,0,0],
#     'Kings XI Punjab': [0, 0, 1, 0, 0, 0, 0, 0,0,0],
#     'Kolkata Knight Riders': [0, 0, 0, 1, 0, 0, 0, 0,0,0],
#     'Mumbai Indians': [0, 0, 0, 0, 1, 0, 0, 0,0,0],
#     'Rajasthan Royals': [0, 0, 0, 0, 0, 1, 0, 0,0,0],
#     'Royal Challengers Bangalore': [0, 0, 0, 0, 0, 0, 1, 0,0,0],
#     'Sunrisers Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1,0,0],
#     'Lucknow Super Giants': [0, 0, 0, 0, 0, 0, 0, 0,1,0],
#     'Gujrat Lions': [0, 0, 0, 0, 0, 0, 0, 0,0,1]
# }

# prediction_array += team_mapping[batting_team]

# # Select Bowling Team
# bowling_team = st.selectbox('Select the Bowling Team', (
#     'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
#     'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
#     'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Lucknow Super Giants', 'Gujrat Lions'
# ))

# if bowling_team == batting_team:
#     st.error('Bowling and Batting teams should be different')

# # Encode Bowling Team
# prediction_array += team_mapping[bowling_team]

# col1, col2 = st.columns(2)

# # Enter Current Over
# with col1:
#     overs = st.number_input('Enter the Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
#     if overs - math.floor(overs) > 0.5:
#         st.error('Please enter a valid over input (one over contains 6 balls)')

# # Enter Current Runs
# with col2:
#     runs = st.number_input('Enter Current Runs', min_value=0, max_value=354, step=1, format='%i')

# # Wickets Taken
# wickets = st.slider('Enter Wickets Fallen', 0, 9)
# wickets = int(wickets)

# col3, col4 = st.columns(2)

# # Runs in last 5 overs
# with col3:
#     runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs', min_value=0, max_value=runs, step=1, format='%i')

# # Wickets in last 5 overs
# with col4:
#     wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs', min_value=0, max_value=wickets, step=1, format='%i')

# # Append other features to prediction array
# prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]

# # Append pitch conditions
# prediction_array += [
#     venue_info['pitch type'], 
#     venue_info['avg first innings score'], 
#     int(venue_info['spin friendly'] == 'Yes'),  # Converts 'Yes' to 1, 'No' to 0
#     int(venue_info['pace friendly'] == 'Yes')
# ]

# # Convert to numpy array for model prediction
# prediction_array = np.array([prediction_array])

# # Predict Score
# if st.button('Predict Score'):
#     predicted_score = int(round(model.predict(prediction_array)[0]))
    
#     # Display the predicted score
#     st.success(f'PREDICTED MATCH SCORE: {predicted_score}')