import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import tempfile
import random
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="TalentLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(
    135deg,
    #0f172a,
    #111827,
    #1e293b
    );
    color:white;
}

section[data-testid="stSidebar"]{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    border-right:1px solid rgba(255,255,255,0.08);
}

.hero{
    background: linear-gradient(
    135deg,
    rgba(37,99,235,0.95),
    rgba(124,58,237,0.9)
    );
    padding:50px;
    border-radius:28px;
    color:white;
    margin-bottom:30px;
    box-shadow:0 10px 40px rgba(0,0,0,0.3);
    animation: glow 6s infinite alternate;
}

@keyframes glow{
    from{
        box-shadow:0 0 20px rgba(37,99,235,0.4);
    }
    to{
        box-shadow:0 0 50px rgba(124,58,237,0.7);
    }
}

.glass-card{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-radius:22px;
    padding:24px;
    margin-bottom:20px;
    border:1px solid rgba(255,255,255,0.08);
}

.skill-tag{
    display:inline-block;
    padding:8px 14px;
    margin:4px;
    border-radius:18px;
    background:#2563eb;
    color:white;
    font-size:13px;
}

.metric-card{
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:18px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.06);
}

h1,h2,h3{
    color:white !important;
}

.small-text{
    color:#d1d5db;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("TalentLens AI")

st.sidebar.markdown("""
### Talent Intelligence Suite

- Resume Parsing
- ATS Ranking
- AI Recommendations
- Skill Heatmaps
- Recruiter Analytics
- Candidate Insights
""")

dark_mode = st.sidebar.toggle("Dark Mode", value=True)

search_candidate = st.sidebar.text_input("Search Candidate")

min_score = st.sidebar.slider(
    "Minimum Match Score",
    0,
    100,
    40
)

st.sidebar.markdown("---")
st.sidebar.success("Developed by Ritu Srivastava")

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="hero">

<h1>TalentLens AI</h1>

<h2>AI-Powered Resume Intelligence Platform</h2>

<p style="font-size:20px;">
Advanced candidate ranking, semantic resume analysis,
skill-gap identification and recruiter recommendations.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Loading AI Engine..."):
    model = load_model()

# ---------------------------------------------------
# PDF TEXT EXTRACTOR
# ---------------------------------------------------

def extract_text(pdf_file):
    text = ""

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:
        try:
            text += page.extract_text()
        except:
            pass

    return text

# ---------------------------------------------------
# AI NOTES
# ---------------------------------------------------

def recruiter_note(score):

    if score >= 85:
        return "Exceptional candidate with strong technical alignment."

    elif score >= 70:
        return "Strong profile with good potential for shortlisting."

    elif score >= 55:
        return "Moderate fit. Consider for secondary screening."

    else:
        return "Low alignment with current job requirements."

# ---------------------------------------------------
# ATS SCORE
# ---------------------------------------------------

def ats_score(resume_text):

    keywords = [
        "python",
        "sql",
        "machine learning",
        "deep learning",
        "tensorflow",
        "aws",
        "nlp",
        "streamlit",
        "power bi",
        "data science"
    ]

    score = 0

    for skill in keywords:
        if skill.lower() in resume_text.lower():
            score += 10

    return min(score,100)

# ---------------------------------------------------
# FILE UPLOADS
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    jd_file = st.file_uploader(
        "Upload Job Description PDF",
        type=["pdf"]
    )

with col2:
    resume_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

# ---------------------------------------------------
# PROCESS
# ---------------------------------------------------

if jd_file and resume_files:

    progress = st.progress(0)

    messages = [
        "Parsing Job Description...",
        "Extracting Resume Skills...",
        "Running Semantic Matching...",
        "Generating AI Rankings...",
        "Preparing Dashboard..."
    ]

    for i in range(len(messages)):
        st.info(messages[i])
        progress.progress((i+1)*20)
        time.sleep(0.5)

    jd_text = extract_text(jd_file)

    jd_embedding = model.encode([jd_text])

    results = []

    skill_database = [
        "Python","SQL","AWS","TensorFlow",
        "Machine Learning","Deep Learning",
        "NLP","Power BI","Tableau","Streamlit"
    ]

    for resume in resume_files:

        resume_text = extract_text(resume)

        resume_embedding = model.encode([resume_text])

        similarity = cosine_similarity(
            jd_embedding,
            resume_embedding
        )[0][0]

        similarity_percentage = round(similarity * 100,2)

        ats = ats_score(resume_text)

        overall = round(
            (similarity_percentage * 0.7) +
            (ats * 0.3),
            2
        )

        matched_skills = []

        missing_skills = []

        for skill in skill_database:

            if skill.lower() in resume_text.lower():
                matched_skills.append(skill)

            else:
                missing_skills.append(skill)

        note = recruiter_note(overall)

        recommendation = (
            "Hire"
            if overall >= 75
            else "Consider"
            if overall >= 55
            else "Reject"
        )

        results.append({
            "Candidate": resume.name,
            "Match Percentage": similarity_percentage,
            "ATS Score": ats,
            "Overall Score": overall,
            "Recommendation": recommendation,
            "Matched Skills": matched_skills,
            "Missing Skills": missing_skills,
            "Recruiter Note": note
        })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Overall Score",
        ascending=False
    ).reset_index(drop=True)

    df["Rank"] = range(1,len(df)+1)

    if search_candidate:
        df = df[
            df["Candidate"]
            .str.contains(search_candidate, case=False)
        ]

    df = df[df["Overall Score"] >= min_score]

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    st.markdown("## Talent Intelligence Dashboard")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
        <h2>{len(df)}</h2>
        <p>Total Candidates</p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        top_score = (
            df["Overall Score"].max()
            if len(df)>0 else 0
        )

        st.markdown(f"""
        <div class="metric-card">
        <h2>{top_score}%</h2>
        <p>Highest Score</p>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        hires = len(df[df["Recommendation"]=="Hire"])

        st.markdown(f"""
        <div class="metric-card">
        <h2>{hires}</h2>
        <p>Recommended Hires</p>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        avg = (
            round(df["Overall Score"].mean(),2)
            if len(df)>0 else 0
        )

        st.markdown(f"""
        <div class="metric-card">
        <h2>{avg}%</h2>
        <p>Average Match</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # CANDIDATE CARDS
    # ---------------------------------------------------

    st.markdown("## Candidate Intelligence")

    for _, row in df.iterrows():

        st.markdown(f"""
        <div class="glass-card">

        <h3>👤 {row['Candidate']}</h3>

        <p>
        <b>Overall Score:</b> {row['Overall Score']}%
        </p>

        <p>
        <b>ATS Score:</b> {row['ATS Score']}%
        </p>

        <p>
        <b>Recommendation:</b> {row['Recommendation']}
        </p>

        <p>
        <b>Recruiter Insight:</b>
        {row['Recruiter Note']}
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Matched Skills")

        for skill in row["Matched Skills"]:
            st.markdown(
                f"<span class='skill-tag'>{skill}</span>",
                unsafe_allow_html=True
            )

        st.markdown("### Missing Skills")

        for skill in row["Missing Skills"]:
            st.markdown(
                f"<span class='skill-tag' style='background:#dc2626'>{skill}</span>",
                unsafe_allow_html=True
            )

        st.markdown("---")

    # ---------------------------------------------------
    # CHARTS
    # ---------------------------------------------------

    st.markdown("## Analytics Dashboard")

    fig = px.bar(
        df,
        x="Candidate",
        y="Overall Score",
        color="Recommendation",
        title="Candidate Ranking Overview"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # HEATMAP
    # ---------------------------------------------------

    heatmap_data = []

    for _, row in df.iterrows():

        for skill in row["Matched Skills"]:

            heatmap_data.append({
                "Candidate": row["Candidate"],
                "Skill": skill,
                "Value": 1
            })

    heat_df = pd.DataFrame(heatmap_data)

    if len(heat_df)>0:

        pivot = heat_df.pivot_table(
            index="Candidate",
            columns="Skill",
            values="Value",
            fill_value=0
        )

        heatmap = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            title="Skill Intelligence Heatmap",
            color_continuous_scale="Blues"
        )

        heatmap.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    # ---------------------------------------------------
    # FINAL TABLE
    # ---------------------------------------------------

    st.markdown("## Final Talent Rankings")

    st.dataframe(
        df,
        use_container_width=True
    )

else:

    st.info(
        "Upload Job Description and Resume PDFs to begin analysis."
    )
