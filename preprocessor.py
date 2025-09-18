import pandas as pd
import re

# Define the preprocess function
def preprocess(data):
    # Define the pattern to extract messages and dates
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm)\s-\s)'

    # Extract messages and dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Ensure messages are aligned correctly
    messages = [messages[i] for i in range(1, len(messages), 2)]

    if len(messages) != len(dates):
        print("Error: Messages and dates lists are not of the same length.")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean and convert 'message_date' safely
    df['message_date'] = df['message_date'].astype(str).str.replace('\u202f', ' ').str.strip()
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p -', errors='coerce')

    # Drop rows with invalid dates (optional)
    df = df.dropna(subset=['message_date'])

    # Rename column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract users and messages
    users = []
    messages_cleaned = []

    for message in df['user_message']:
        entry = re.split(r'^([\w\W]+?):\s', message)
        if len(entry) > 2:
            users.append(entry[1])
            messages_cleaned.append(entry[2])
        else:
            users.append('group_notification')
            messages_cleaned.append(entry[0].strip())

    df['users'] = users
    df['message'] = messages_cleaned
    df.drop(columns=['user_message'], inplace=True)

    # Add date-related columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df
