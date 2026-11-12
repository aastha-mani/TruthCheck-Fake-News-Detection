import streamlit as st
import joblib
import requests
import re
from collections import Counter
from urllib.parse import urlparse, quote_plus
import xml.etree.ElementTree as ET

# Page configuration
st.set_page_config(
    page_title="TruthCheck | Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load trained model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Custom styling (kept from your existing interface)
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}

.main-title {
    text-align: center;
    color: #173b70;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #52647a;
    font-size: 17px;
}

.stTextArea textarea {
    border: 2px solid #b8cbe5;
    border-radius: 10px;
    font-size: 16px;
}

.stButton button {
    background-color: #1d5da8;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #123f78;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# Improved live-news search
STOP_WORDS = {
    "the", "and", "for", "that", "with", "this", "from", "was", "were",
    "are", "has", "have", "had", "will", "would", "could", "should",
    "into", "about", "after", "before", "their", "there", "they", "them",
    "then", "than", "when", "where", "which", "while", "who", "what",
    "how", "why", "its", "his", "her", "our", "your", "you", "also",
    "said", "says", "according", "reported", "report", "news", "article",
    "more", "most", "some", "many", "such", "been", "being", "over",
    "under", "between", "through", "during", "because", "not", "but",
    "all", "any", "can", "may", "one", "two", "new", "latest",
    "union", "cabinet", "government", "india", "indian"
}

def make_search_queries(article):
    """
    Build up to two targeted queries.
    Prefer named organisations, places and products over frequent words.
    """
    cleaned = re.sub(r"\s+", " ", article).strip()

    # Find capitalized names/phrases such as "Noida Metro Aqua Line"
    proper_phrases = re.findall(
        r"\b[A-Z][a-zA-Z0-9&'-]*(?:\s+[A-Z][a-zA-Z0-9&'-]*){0,3}\b",
        cleaned
    )

    useful_phrases = []
    seen = set()
    for phrase in proper_phrases:
        phrase = phrase.strip(" ,.")
        if len(phrase) < 3:
            continue
        if phrase.lower() in STOP_WORDS:
            continue
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            useful_phrases.append(phrase)

    # First choice: a specific named phrase or two.
    queries = []
    if len(useful_phrases) >= 2:
        queries.append(f'"{useful_phrases[-2]}" "{useful_phrases[-1]}"')
    elif useful_phrases:
        queries.append(f'"{useful_phrases[-1]}"')

    # Fallback: use meaningful words, not just the most repeated words.
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", cleaned)
    keywords = [
        word for word in words
        if word.lower() not in STOP_WORDS and not word.isdigit()
    ]
    counts = Counter(word.lower() for word in keywords)
    ranked = [word for word, _ in counts.most_common(7)]
    if len(ranked) >= 2:
        fallback = " ".join(ranked[:5])
        if fallback.lower() not in [q.lower() for q in queries]:
            queries.append(fallback)

    return queries[:2]


@st.cache_data(ttl=900, show_spinner=False)
def get_gdelt_articles(query):
    """Cached GDELT request. Cache lasts 15 minutes to reduce rate limiting."""
    api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 8,
        "timespan": "1year",
        "sort": "datedesc"
    }
    headers = {"User-Agent": "TruthCheckStudentProject/2.0"}

    # One short retry for temporary rate limiting or server errors.
    for attempt in range(2):
        try:
            response = requests.get(
                api_url, params=params, headers=headers, timeout=20
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 0:
                    import time
                    time.sleep(3)
                    continue
            response.raise_for_status()

            data = response.json()
            articles = data.get("articles", [])
            cleaned = []
            seen_urls = set()

            for item in articles:
                title = (item.get("title") or "").strip()
                url = (item.get("url") or "").strip()
                if title and url and url not in seen_urls:
                    seen_urls.add(url)
                    cleaned.append({
                        "title": title,
                        "url": url,
                        "domain": item.get("domain") or urlparse(url).netloc,
                        "date": item.get("seendate", "Date not supplied")
                    })
            return cleaned, None

        except requests.exceptions.Timeout:
            return [], "The live news search timed out. Please try again."
        except requests.exceptions.HTTPError as error:
            if response.status_code == 429:
                return [], (
                    "The live news service is temporarily rate-limiting requests. "
                    "Wait a few minutes and try again. Your ML prediction is still available."
                )
            return [], f"The live news service returned an error: {error}"
        except requests.exceptions.RequestException as error:
            return [], f"Could not connect to the live news service: {error}"
        except ValueError:
            return [], "The news service returned an unreadable response."
        except Exception as error:
            return [], f"Live search could not be completed: {error}"

    return [], "The live news service is temporarily unavailable. Please try again later."


def search_google_news_rss(query):
    """Fallback coverage search through Google News RSS; no API key required."""
    rss_url = "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=en-IN&gl=IN&ceid=IN:en"
    response = requests.get(
        rss_url,
        headers={"User-Agent": "Mozilla/5.0 TruthCheckStudentProject/2.1"},
        timeout=12
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    results = []
    seen = set()

    for item in root.findall(".//item")[:6]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "Date not supplied").strip()
        if title and link and link not in seen:
            seen.add(link)
            results.append({
                "title": title,
                "url": link,
                "domain": urlparse(link).netloc or "Google News",
                "date": pub_date
            })
    return results


def search_live_news(article):
    queries = make_search_queries(article)
    if not queries:
        return [], "The article did not contain enough searchable keywords."

    all_articles = []
    seen_urls = set()
    service_errors = []

    # First try GDELT. If it is unavailable or returns no matches,
    # use Google News RSS as a second independent coverage source.
    for query in queries:
        articles, error = get_gdelt_articles(query)
        if error:
            service_errors.append("GDELT: " + error)
            if "rate-limiting" in error.lower():
                break
            continue
        for item in articles:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_articles.append(item)
        if all_articles:
            break

    if not all_articles:
        for query in queries:
            try:
                articles = search_google_news_rss(query)
                for item in articles:
                    if item["url"] not in seen_urls:
                        seen_urls.add(item["url"])
                        all_articles.append(item)
                if all_articles:
                    break
            except requests.exceptions.RequestException as error:
                service_errors.append("Google News: " + str(error))
            except ET.ParseError:
                service_errors.append("Google News returned an unreadable feed.")
            except Exception as error:
                service_errors.append("Google News: " + str(error))

    if all_articles:
        return all_articles[:10], None
    if service_errors:
        return [], (
            "Live news search services could not return matching coverage. "
            "Please try again later. " + service_errors[0]
        )
    return [], None


# Application heading
st.markdown(
    '<div class="main-title">📰 TruthCheck</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter a news article below. "
    "Our machine learning model will analyze its text."
)

# News input (same label, height and placeholder)
news = st.text_area(
    "Paste Your News Article Here:",
    height=220,
    placeholder="Example: The government has announced a new education policy..."
)

# Prediction and live search use the same existing button
if st.button("🔍 Analyze News", use_container_width=True):

    if news.strip():

        transformed_news = vectorizer.transform([news])
        prediction = model.predict(transformed_news)

        st.divider()
        st.subheader("Analysis Result")

        if prediction[0] == 1:
            st.success("✅ ML Prediction: This news is classified as REAL.")
        else:
            st.error("⚠️ ML Prediction: This news is classified as FAKE.")

        st.caption(
            "Prediction generated using a trained Logistic Regression model."
        )

        # Live news coverage lookup
        st.divider()
        st.subheader("Live News Coverage")

        with st.spinner("Searching recent online news coverage..."):
            live_articles, search_error = search_live_news(news)

        if search_error:
            st.warning(search_error)
            st.caption(
                "The machine-learning prediction above is still available. "
                "Live search could not be completed."
            )
        elif live_articles:
            st.info(
                f"Found {len(live_articles)} potentially related article(s) in GDELT's "
                "news coverage from the past year."
            )

            for index, article in enumerate(live_articles, start=1):
                st.markdown(f"**{index}. [{article['title']}]({article['url']})**")
                st.caption(
                    f"Source: {article['domain']} | "
                    f"Date seen: {article['date']}"
                )

            st.warning(
                "Matching coverage is not proof that the submitted claim is true. "
                "Open the sources and compare the details, dates and original reporting."
            )
        else:
            st.warning(
                "No matching articles were found in the GDELT search. "
                "This does not prove the news is fake. The story may be local, "
                "not indexed, or described using different words."
            )

    else:
        st.warning("Please enter a news article before analyzing.")

st.divider()

st.caption(
    "TruthCheck | Machine Learning Project | "
    "Educational Demonstration"
)

st.caption(
    "Disclaimer: Predictions are based on learned text patterns "
    "and should not be treated as verified facts."
)
st.caption(
    "Note: Live coverage checks GDELT first and Google News RSS as a fallback. "
    "Search results are leads for checking, not proof of truth or falsity."
)
