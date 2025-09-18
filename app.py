import streamlit as st
import helper
from preprocessor import preprocess
import pandas as pd

# Page config
st.set_page_config(layout="wide")
st.sidebar.title("📊 WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("📁 Upload your WhatsApp chat (.txt)")

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocess(data)

        # ===== PREPROCESSING: SAFETY AND TYPE CHECKS =====
        required_cols = {'users', 'message', 'date'}
        if not required_cols.issubset(df.columns):
            st.error("❌ Uploaded file is missing required columns: 'users', 'message', or 'date'.")
            st.stop()

        df.dropna(subset=['message', 'date'], inplace=True)
        df['message'] = df['message'].astype(str)
        df['users'] = df['users'].fillna('Unknown').astype(str)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        # Sidebar - Select user
        st.sidebar.subheader("🔍 Analysis Options")
        user_list = df['users'].unique().tolist()
        selected_user = st.sidebar.selectbox("Select User", ['OVERALL'] + sorted(user_list))

        # Title
        st.title("📈 WhatsApp Chat Analyzer")
        st.markdown("---")

        # Filtered data
        user_df = df if selected_user == 'OVERALL' else df[df['users'] == selected_user]

        # ===== CHECK FOR EMPTY USER DATA =====
        if user_df.empty:
            st.warning(f"No messages found for user: **{selected_user}**")
            st.stop()

        # ===== BASIC STATS =====
        st.header("📊 Basic Statistics")
        num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", num_words)
        col3.metric("Media Shared", num_media)
        col4.metric("Links Shared", num_links)
        st.markdown("---")

        # ===== EXPANDER FOR DETAILED ANALYSIS =====
        with st.expander("🧠 Detailed Analysis"):
            st.subheader("👥 Most Busy Users")
            helper.most_busy_users(df)
            st.markdown("---")

            st.subheader("📌 User Contributions")
            contributions_df = helper.user_contributions(df)
            st.dataframe(contributions_df)
            st.markdown("---")

            st.subheader("☁️ Word Cloud")
            wordcloud = helper.generate_wordcloud(selected_user, df)
            st.image(wordcloud.to_array(), use_column_width=True)
            st.markdown("---")

            st.subheader("🗣️ Most Common Words")
            common_words_df = helper.most_common_words(selected_user, df)
            helper.plot_most_common_words(common_words_df)
            st.dataframe(common_words_df)
            st.markdown("---")

            st.subheader("😀 Emoji Analysis")
            emoji_df = helper.emoji_analysis(selected_user, df)
            st.dataframe(emoji_df)
            st.markdown("---")

            st.subheader("📆 Message Timeline")
            helper.timeline_analysis(selected_user, df)
            st.markdown("---")

            st.subheader("⏰ Daily Activity")
            helper.daily_activity_analysis(selected_user, df)
            st.markdown("---")

            st.subheader("📅 Weekly Activity")
            helper.weekly_activity_analysis(selected_user, df)
            st.markdown("---")

            st.subheader("💬 Sentiment Analysis")
            helper.sentiment_plot(selected_user, df)
            st.markdown("---")

            st.subheader("🔥 Hourly Activity Heatmap")
            helper.hourly_activity_heatmap(selected_user, df)
            st.markdown("---")

            st.subheader("📍 Most Active Days & Hours")
            most_active_day, most_active_hour = helper.most_active_days_hours(selected_user, df)
            st.write(f"**Most Active Day:** {most_active_day}")
            st.write(f"**Most Active Hour:** {most_active_hour}")

    except Exception as e:
        st.error(f"⚠️ Something went wrong while processing the file: {e}")

else:
    st.info("📤 Please upload a WhatsApp chat file (.txt) to begin analysis.")
