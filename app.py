import streamlit as st
import fitz  # PyMuPDF
import re
import math
import io
from collections import Counter

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resume–JD Matcher",
    page_icon="🎯",
    layout="wide",
)

# ─────────────────────────────────────────────
# STOP WORDS (lightweight, no NLTK download)
# ─────────────────────────────────────────────
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn","worked","looking","candidates","skilled",
    "seeking","experience","years","including","using","used","proficient",
    "knowledge","ability","strong","good","excellent","required","preferred",
    "responsibilities","role","position","team","company","opportunity",
    "apply","please","must","also","well","new","work","project","projects",
    "ability","able","ensure","help","support","across","within","provide",
}

# Simple suffix rules for basic stemming
def simple_stem(word):
    suffixes = ["ing","tion","tions","ment","ments","ness","ful","less","ive","ed","er","est","ly","al","ic","ous","ize","ise"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)]
    return word

# ─────────────────────────────────────────────
# NLP PIPELINE
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    """PDF parsing using PyMuPDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    """Lowercase + remove special symbols."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    """Split into word tokens."""
    return text.split()

def remove_stopwords(tokens):
    """Filter out stop words."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def stem_tokens(tokens):
    """Apply simple suffix stemming."""
    return [simple_stem(t) for t in tokens]

def full_pipeline(text):
    """End-to-end NLP pipeline → list of processed tokens."""
    cleaned   = clean_text(text)
    tokens    = tokenize(cleaned)
    no_stop   = remove_stopwords(tokens)
    stemmed   = stem_tokens(no_stop)
    return stemmed, no_stop  # return both for display purposes

# ─────────────────────────────────────────────
# TF-IDF & COSINE SIMILARITY
# ─────────────────────────────────────────────
def compute_tf(tokens):
    """Term Frequency for a token list."""
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: freq / total for term, freq in count.items()}

def compute_idf(doc_tokens_list):
    """IDF across a small corpus (resume + JD)."""
    N = len(doc_tokens_list)
    all_terms = set(t for tokens in doc_tokens_list for t in tokens)
    idf = {}
    for term in all_terms:
        df = sum(1 for tokens in doc_tokens_list if term in tokens)
        idf[term] = math.log(N / df) if df else 0
    return idf

def tfidf_vector(tf, idf, vocab):
    """Build TF-IDF vector over shared vocab."""
    return [tf.get(term, 0) * idf.get(term, 0) for term in vocab]

def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors."""
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

# ─────────────────────────────────────────────
# SKILL / KEYWORD MATCHING
# ─────────────────────────────────────────────
TECH_SKILLS = {
    "python","java","javascript","typescript","c","cpp","go","rust","scala","kotlin",
    "sql","nosql","mongodb","postgresql","mysql","redis","cassandra",
    "react","angular","vue","nodejs","django","flask","fastapi","spring",
    "aws","gcp","azure","ec2","s3","lambda","kubernetes","docker","terraform",
    "pytorch","tensorflow","sklearn","scikit","pandas","numpy","scipy","matplotlib",
    "cnn","rnn","lstm","transformer","bert","gpt","nlp","cv","ml","ai","dl",
    "git","linux","bash","rest","api","microservices","agile","scrum",
    "spark","hadoop","kafka","airflow","dbt","snowflake","bigquery",
    "tableau","powerbi","excel","looker",
}

def extract_skills(tokens):
    """Return skill keywords found in tokens."""
    token_set = set(tokens)
    return sorted(token_set & TECH_SKILLS)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (for ML decision model)
# ─────────────────────────────────────────────
def build_feature_vector(similarity, resume_skills, jd_skills):
    matched   = set(resume_skills) & set(jd_skills)
    missing   = set(jd_skills) - set(resume_skills)
    extra     = set(resume_skills) - set(jd_skills)

    features = {
        "cosine_similarity": round(similarity, 4),
        "matched_skills"   : len(matched),
        "missing_skills"   : len(missing),
        "extra_skills"     : len(extra),
        "jd_coverage_%"    : round(len(matched) / max(len(jd_skills), 1) * 100, 1),
    }
    return features, matched, missing, extra

# ─────────────────────────────────────────────
# SIMPLE LOGISTIC-STYLE SCORING (no sklearn dep)
# ─────────────────────────────────────────────
def predict_selection_probability(features):
    """
    Weighted linear combination → sigmoid.
    Weights calibrated to match example in doc (≈0.78 for good match).
    """
    score = (
          0.50 * features["cosine_similarity"]
        + 0.03 * features["matched_skills"]
        - 0.02 * features["missing_skills"]
        + 0.004 * features["extra_skills"]
        + 0.003 * features["jd_coverage_%"]
    )
    # Sigmoid
    prob = 1 / (1 + math.exp(-10 * (score - 0.35)))
    return round(prob, 4)

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("🎯 Resume–JD Matcher")
st.markdown(
    "Upload a **Resume PDF** and paste a **Job Description** to get an instant NLP-powered match score."
)

# Sidebar – pipeline overview
with st.sidebar:
    st.header("⚙️ NLP Pipeline")
    stages = [
        ("📄", "Text Extraction",    "PyMuPDF PDF parsing"),
        ("🧹", "Text Cleaning",      "Lowercase, remove symbols"),
        ("🔇", "Stop Word Removal",  "Filter filler words"),
        ("✂️", "Tokenization",       "Split into tokens"),
        ("🌿", "Stemming",           "Reduce to root forms"),
        ("🏷️", "Skill Extraction",   "NER-style keyword match"),
        ("📐", "TF-IDF",             "Weight term importance"),
        ("📏", "Cosine Similarity",  "Vector matching score"),
        ("🤖", "ML Decision Model",  "Selection probability"),
    ]
    for icon, name, desc in stages:
        st.markdown(f"**{icon} {name}**  \n<small>{desc}</small>", unsafe_allow_html=True)
        st.markdown("---")

# ─── Input columns ───
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume")

with col2:
    st.subheader("📋 Job Description")
    jd_text_input = st.text_area(
        "Paste JD here",
        height=250,
        placeholder="Looking for candidates skilled in Python, ML, AWS...",
    )

# ─── Run Analysis ───
run_btn = st.button("🚀 Analyse Match", use_container_width=True, type="primary")

if run_btn:
    if not resume_file:
        st.error("Please upload a Resume PDF.")
    elif not jd_text_input.strip():
        st.error("Please paste a Job Description.")
    else:
        with st.spinner("Running NLP pipeline…"):

            # ── 1. Text Extraction ──
            resume_raw = extract_text_from_pdf(resume_file)
            jd_raw     = jd_text_input

            # ── 2-5. NLP Pipeline ──
            resume_tokens, resume_no_stop = full_pipeline(resume_raw)
            jd_tokens,     jd_no_stop     = full_pipeline(jd_raw)

            # ── 6. Skill Extraction ──
            resume_skills = extract_skills(resume_tokens)
            jd_skills     = extract_skills(jd_tokens)

            # ── 7. TF-IDF ──
            resume_tf = compute_tf(resume_tokens)
            jd_tf     = compute_tf(jd_tokens)
            idf       = compute_idf([resume_tokens, jd_tokens])
            vocab     = sorted(set(resume_tokens) | set(jd_tokens))

            r_vec = tfidf_vector(resume_tf, idf, vocab)
            j_vec = tfidf_vector(jd_tf,     idf, vocab)

            # ── 8. Cosine Similarity ──
            similarity = cosine_similarity(r_vec, j_vec)

            # ── 9. Feature Engineering + Score ──
            features, matched, missing, extra = build_feature_vector(
                similarity, resume_skills, jd_skills
            )
            prob = predict_selection_probability(features)

        # ─────────────────────────────────────────
        # RESULTS
        # ─────────────────────────────────────────
        st.markdown("---")
        st.header("📊 Results")

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cosine Similarity",     f"{similarity:.2%}")
        m2.metric("Selection Probability", f"{prob:.2%}")
        m3.metric("Skills Matched",        f"{len(matched)} / {len(jd_skills)}")
        m4.metric("JD Coverage",           f"{features['jd_coverage_%']}%")

        # Similarity gauge bar
        st.markdown("### 🔍 Match Score")
        color = "🟢" if similarity > 0.6 else ("🟡" if similarity > 0.35 else "🔴")
        st.progress(min(similarity, 1.0))
        st.markdown(f"{color} **{similarity:.2%}** overall TF-IDF cosine similarity")

        # Skills breakdown
        st.markdown("### 🏷️ Skills Breakdown")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.success(f"✅ **Matched ({len(matched)})**")
            for s in sorted(matched):
                st.markdown(f"- `{s}`")
        with sc2:
            st.error(f"❌ **Missing from Resume ({len(missing)})**")
            for s in sorted(missing):
                st.markdown(f"- `{s}`")
        with sc3:
            st.info(f"➕ **Extra in Resume ({len(extra)})**")
            for s in sorted(extra):
                st.markdown(f"- `{s}`")

        # Feature vector
        st.markdown("### 📐 Feature Vector (ML Input)")
        fv_cols = st.columns(len(features))
        for col, (k, v) in zip(fv_cols, features.items()):
            col.metric(k.replace("_", " ").title(), v)

        # Pipeline walkthrough
        with st.expander("🔬 Pipeline Detail – Tokens & Processing"):
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("**Resume tokens (after cleaning & stop-word removal)**")
                st.write(resume_no_stop[:60])
            with t2:
                st.markdown("**JD tokens (after cleaning & stop-word removal)**")
                st.write(jd_no_stop[:60])

        # Interpretation table
        st.markdown("### 🗺️ Pipeline Summary")
        pipeline_data = {
            "Stage": ["Cleaning","Tokenization","NLP / NER","TF-IDF","Cosine Similarity","ML Model"],
            "Purpose": [
                "Remove noise & lowercase",
                "Break text into tokens",
                "Extract skill keywords",
                "Weight term importance",
                "Matching score",
                "Selection probability",
            ],
            "Result": [
                "✅ Done",
                f"{len(resume_tokens)} resume / {len(jd_tokens)} JD tokens",
                f"{len(resume_skills)} resume skills / {len(jd_skills)} JD skills",
                f"Vocab size: {len(vocab)}",
                f"{similarity:.4f}",
                f"{prob:.4f}",
            ],
        }
        st.table(pipeline_data)

        # Verdict
        st.markdown("---")
        if prob >= 0.70:
            st.success(f"🏆 **Strong Match!** Selection probability: **{prob:.1%}** — Your resume aligns well with this JD.")
        elif prob >= 0.45:
            st.warning(f"⚠️ **Moderate Match.** Selection probability: **{prob:.1%}** — Consider adding the missing skills.")
        else:
            st.error(f"❌ **Weak Match.** Selection probability: **{prob:.1%}** — Significant skill gaps detected.")
