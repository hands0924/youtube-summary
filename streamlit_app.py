import streamlit as st
import requests
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


# Function to fetch data from SERP API
def fetch_youtube_data(api_key, query):
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": api_key
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    return data


# Function to summarize YouTube video using LangChain
def summarize_youtube_video(video_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url=video_url,language='ko')
    documents = loader.load()
    # Use RecursiveCharacterTextSplitter if the document is too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    key = "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"
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key = key)

    # Load the summarize chain
    summaries = []
    for text in texts:
        response = llm([HumanMessage(content=f"다음 텍스트를 한국어로 요약해주세요:\n\n{text}")])
        summaries.append(response.content)

    final_summary = llm([HumanMessage(content=f"다음 요약들을 종합하여 전체 내용을 한국어로 요약해주세요:\n\n{''.join(summaries)}")])
    result = final_summary.content
    return result



# Streamlit app
def main():
    st.title("YouTube Video Search and Summarize App")

    api_key = st.text_input("Enter your SERP API key", type="password")
    search_query = st.text_input("Enter search keyword")
    num_results = st.selectbox("Select number of results to display", [5, 10, 15, 20], index=0)

    if st.button("Search") and api_key and search_query:
        st.write("Searching for videos...")
        data = fetch_youtube_data(api_key, search_query)

        if "video_results" in data:
            videos = data["video_results"][:num_results]
            for index, video in enumerate(videos):
                st.image(video["thumbnail"]["static"])
                st.write(f"**Title**: {video['title']}")
                st.write(f"**Link**: [Watch on YouTube]({video['link']})")
                st.write(f"**Published Date**: {video['published_date']}")
                st.write(f"**Views**: {video['views']}")
                st.write(f"**Description**: {video['description']}")

                # Add button to summarize video
                if st.button(f"Summarize Video {index + 1}"):
                    with st.spinner('Summarizing...'):
                        summary = summarize_youtube_video(video['link'])
                        st.write("**Summary**: ")
                        st.write(summary)

                st.write("---")
        else:
            st.write("No videos found.")


if __name__ == "__main__":
    main()
