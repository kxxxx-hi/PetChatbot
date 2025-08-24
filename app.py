import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Pet Adoption Chatbot", page_icon="🐾", layout="wide")

INTENTS = ["find_pet", "adoption_info", "pet_details"]

@st.cache_data
def load_listings(path="pets.csv"):
    df = pd.read_csv(path)
    for c in ["type","breed","age","sex","size","color","location","description","name"]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str)
    if "appeal_score" not in df.columns: df["appeal_score"] = 0.5
    return df

listings = load_listings()

@st.cache_resource
def build_intent_clf():
    # seed examples; replace with your labeled set
    X = [
        "show me a small dog in singapore",
        "find a kitten near jurong east",
        "i want a friendly medium size dog",
        "how do i adopt a pet",
        "what documents do i need to adopt",
        "tell me the adoption steps",
        "details about bella",
        "more info about pet d001",
        "show pictures of coco",
    ]
    y = ["find_pet","find_pet","find_pet","adoption_info","adoption_info","adoption_info","pet_details","pet_details","pet_details"]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xv  = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=1000).fit(Xv, y)
    return vec, clf

VEC, INTENT_CLF = build_intent_clf()

def classify_intent(text: str) -> str:
    return INTENT_CLF.predict(VEC.transform([text]))[0]

TYPE_WORDS = {"dog":"dog","dogs":"dog","puppy":"dog","cat":"cat","cats":"cat","kitten":"cat"}
SIZE_WORDS = {"small":"small","smol":"small","medium":"medium","large":"large","big":"large"}
AGE_WORDS  = {"puppy":"young","kitten":"young","young":"young","adult":"adult","senior":"senior"}
SEX_WORDS  = {"male":"male","female":"female","boy":"male","girl":"female"}
BREEDS     = sorted(listings["breed"].unique())[:500]

def extract_entities(text: str) -> dict:
    t = text.lower()
    ent = {"type":None,"breed":None,"size":None,"age":None,"sex":None,"location":None}
    for w,v in TYPE_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t): ent["type"]=v
    for w,v in SIZE_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t): ent["size"]=v
    for w,v in AGE_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t): ent["age"]=v
    for w,v in SEX_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t): ent["sex"]=v

    best = process.extractOne(t, BREEDS, scorer=fuzz.partial_ratio)
    if best and best[1] >= 85: ent["breed"] = best[0]

    # naive location
    for loc in ["Singapore","Jurong","Woodlands","Tampines","Hougang","Yishun","Clementi","Bedok","Toa Payoh","Jurong East"]:
        if re.search(rf"\b{loc.lower()}\b", t): ent["location"] = loc
    return ent

@st.cache_resource
def get_vader():
    try:
        return SentimentIntensityAnalyzer()
    except:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()

SIA = get_vader()

def appeal_from_text(desc: str) -> float:
    if not desc: return 0.5
    s = SIA.polarity_scores(desc)["compound"]
    return 0.5*(s+1.0)

def match_and_rank(query: str, ents: dict, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    work = df.copy()
    if "appeal_score" in work and (work["appeal_score"]==0.5).all():
        work["appeal_score"] = work["description"].fillna("").apply(appeal_from_text)
    if ents["type"]:     work = work[work["type"].str.lower()==ents["type"]]
    if ents["breed"]:    work = work[work["breed"].str.lower()==ents["breed"].lower()]
    if ents["size"]:     work = work[work["size"].str.lower()==ents["size"]]
    if ents["age"]:      work = work[work["age"].str.lower()==ents["age"]]
    if ents["sex"]:      work = work[work["sex"].str.lower()==ents["sex"]]
    if ents["location"]: work = work[work["location"].str.contains(ents["location"], case=False, na=False)]

    if work.empty:
        corpus = (df["name"].fillna("") + " " + df["description"].fillna("")).tolist()
        vec = TfidfVectorizer(min_df=1, stop_words="english")
        X = vec.fit_transform(corpus)
        q = vec.transform([query])
        sim = (X @ q.T).toarray().ravel()
        base = df.copy()
        base["sim"] = sim
        base["score"] = 0.7*base["sim"] + 0.3*base["appeal_score"]
        return base.sort_values("score", ascending=False).head(k)

    q = query.lower()
    work["sim_name"] = work["name"].fillna("").str.lower().apply(lambda s: fuzz.partial_ratio(q, s)/100.0)
    work["sim_desc"] = work["description"].fillna("").str.lower().apply(lambda s: fuzz.token_set_ratio(q, s)/100.0)
    work["score"] = 0.5*work["sim_desc"] + 0.2*work["sim_name"] + 0.3*work["appeal_score"]
    return work.sort_values("score", ascending=False).head(k)

st.title("🐾 Pet Adoption Chatbot")

with st.sidebar:
    st.markdown("### Filters")
    pet_type = st.selectbox("Type", ["", "dog", "cat"])
    size     = st.selectbox("Size", ["", "small","medium","large"])
    age      = st.selectbox("Age", ["", "young","adult","senior"])
    sex      = st.selectbox("Sex", ["", "male","female"])
    location = st.text_input("Location")
    if st.button("Clear chat"): st.session_state.clear()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Tell me what you’re looking for. Example: 'Looking for a small dog in Singapore'."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.write(m["content"])

user = st.chat_input("Type a message")
if user:
    st.session_state.messages.append({"role":"user","content":user})
    with st.chat_message("user"): st.write(user)

    intent = classify_intent(user)
    ents   = extract_entities(user)

    # sidebar overrides
    if pet_type: ents["type"]=pet_type
    if size:     ents["size"]=size
    if age:      ents["age"]=age
    if sex:      ents["sex"]=sex
    if location: ents["location"]=location

    with st.chat_message("assistant"):
        if intent == "adoption_info":
            st.write("Typical flow: submit interest → interview/home-check → fees & documents → pickup/transport. Share your area for shelter-specific steps.")
        elif intent == "pet_details":
            st.write("Provide a pet name or ID (e.g., D001) to fetch details.")
        else:
            hits = match_and_rank(user, ents, listings, k=6)
            if hits.empty:
                st.write("No matches. Try broader criteria or remove filters.")
            else:
                st.write(f"Top results ({len(hits)}):")
                for _, row in hits.iterrows():
                    c1,c2,c3 = st.columns([1,2,2])
                    with c1:
                        if isinstance(row.get("photo_url",""), str) and row["photo_url"].startswith("http"):
                            st.image(row["photo_url"], use_container_width=True)
                        else:
                            st.write("No image")
                    with c2:
                        st.markdown(f"**{row.get('name','Unknown')}**")
                        st.caption(f"{row.get('type','?')} • {row.get('breed','?')} • {row.get('age','?')} • {row.get('sex','?')} • {row.get('size','?')}")
                        st.caption(row.get("location",""))
                    with c3:
                        st.progress(min(max(row.get("appeal_score",0.5),0.0),1.0))
                        with st.expander("Description"):
                            st.write(row.get("description",""))
                st.write("Refine search by adding type/breed/size/age/location in your message.")
    st.session_state.messages.append({"role":"assistant","content":"(results shown above)"})