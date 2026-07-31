"""
skills_db.py
============
A controlled skills taxonomy used by resume_engine.py.

Every skill is stored with:
  - category: one bucket used for the "Skill Categorization" UI
              (Languages, Frameworks, Libraries, Databases, Cloud & DevOps,
              Tools & Platforms, Soft Skills)
  - surface_forms: literal strings/aliases that count as a mention of the
                    skill in raw resume/JD text (exact-match layer)

SKILL_SYNONYMS additionally maps common abbreviations / related phrasings
to a canonical skill key. This isn't a language model -- it's a lightweight
dictionary that lets "ML" recognize "Machine Learning", "K8s" recognize
"Kubernetes", etc. It is used as ONE signal inside the semantic matching
step in resume_engine.py, never presented on its own as "AI matched this".
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Set

CATEGORIES = [
    "Programming",
    "Data Science & ML"
    "Data Analysis & Statistics"
    "Languages",
    "Frameworks",
    "Libraries",
    "Databases",
    "Cloud & DevOps",
    "Tools & Platforms",
    "Soft Skills",
]

SKILLS_DB: Dict[str, Dict] = {
    # ---------------- Languages ----------------
    "python":        {"category": "Languages", "surface_forms": ["python","Python"]},
    "java":          {"category": "Languages", "surface_forms": ["java"]},
    "c++":           {"category": "Languages", "surface_forms": ["c++", "cpp"]},
    "c#":            {"category": "Languages", "surface_forms": ["c#", "c-sharp", "csharp"]},
    "javascript":    {"category": "Languages", "surface_forms": ["javascript", "js"]},
    "typescript":    {"category": "Languages", "surface_forms": ["typescript", "ts"]},
    "go":            {"category": "Languages", "surface_forms": ["golang", " go "]},
    "rust":          {"category": "Languages", "surface_forms": ["rust"]},
    "sql":           {"category": "Languages", "surface_forms": ["sql","Mysql",'mysql',"MySql"]},
    "r":             {"category": "Languages", "surface_forms": [" r programming", "r language"]},
    "php":           {"category": "Languages", "surface_forms": ["php"]},
    "scala":         {"category": "Languages", "surface_forms": ["scala"]},
    "kotlin":        {"category": "Languages", "surface_forms": ["kotlin"]},
    "swift":         {"category": "Languages", "surface_forms": ["swift"]},

    # ---------------- Frameworks ----------------
    "django":        {"category": "Frameworks", "surface_forms": ["django"]},
    "flask":         {"category": "Frameworks", "surface_forms": ["flask"]},
    "fastapi":       {"category": "Frameworks", "surface_forms": ["fastapi"]},
    "spring":        {"category": "Frameworks", "surface_forms": ["spring boot", "spring framework", "spring"]},
    "react":         {"category": "Frameworks", "surface_forms": ["react.js", "reactjs", "react"]},
    "angular":       {"category": "Frameworks", "surface_forms": ["angular"]},
    "vue":           {"category": "Frameworks", "surface_forms": ["vue.js", "vuejs", "vue"]},
    "node.js":       {"category": "Frameworks", "surface_forms": ["node.js", "nodejs", "node"]},
    "express":       {"category": "Frameworks", "surface_forms": ["express.js", "expressjs", "express"]},
    "next.js":       {"category": "Frameworks", "surface_forms": ["next.js", "nextjs"]},
    ".net":          {"category": "Frameworks", "surface_forms": [".net", "dotnet", "asp.net"]},

    # ---------------- Libraries ----------------
    "pandas":        {"category": "Libraries", "surface_forms": ["pandas","Pandas"]},
    "numpy":         {"category": "Libraries", "surface_forms": ["numpy","Numpy"]},
    "scikit-learn":  {"category": "Libraries", "surface_forms": ["scikit-learn", "sklearn","Scikit-Learn"]},
    "tensorflow":    {"category": "Libraries", "surface_forms": ["tensorflow","TensorFlow"]},
    "pytorch":       {"category": "Libraries", "surface_forms": ["pytorch", "torch"]},
    "keras":         {"category": "Libraries", "surface_forms": ["keras","Keras"]},
    "matplotlib":    {"category": "Libraries", "surface_forms": ["matplotlib","Matplotlib"]},
    "opencv":        {"category": "Libraries", "surface_forms": ["opencv","OpenCV"]},
    "spacy":         {"category": "Libraries", "surface_forms": ["spacy"]},
    "nltk":          {"category": "Libraries", "surface_forms": ["nltk"]},

    # ---------------- Databases ----------------
    "mysql":         {"category": "Databases", "surface_forms": ["mysql","SQL","MySQL","Mysql"]},
    "postgresql":    {"category": "Databases", "surface_forms": ["postgresql", "postgres"]},
    "mongodb":       {"category": "Databases", "surface_forms": ["mongodb", "mongo"]},
    "redis":         {"category": "Databases", "surface_forms": ["redis"]},
    "oracle":        {"category": "Databases", "surface_forms": ["oracle db", "oracle"]},
    "sql server":    {"category": "Databases", "surface_forms": ["sql server", "mssql"]},
    "cassandra":     {"category": "Databases", "surface_forms": ["cassandra"]},
    "elasticsearch": {"category": "Databases", "surface_forms": ["elasticsearch", "elastic search"]},
    "dynamodb":      {"category": "Databases", "surface_forms": ["dynamodb"]},

    # ---------------- Cloud & DevOps ----------------
    "aws":           {"category": "Cloud & DevOps", "surface_forms": ["aws", "amazon web services"]},
    "azure":         {"category": "Cloud & DevOps", "surface_forms": ["azure"]},
    "gcp":           {"category": "Cloud & DevOps", "surface_forms": ["gcp", "google cloud"]},
    "docker":        {"category": "Cloud & DevOps", "surface_forms": ["docker"]},
    "kubernetes":    {"category": "Cloud & DevOps", "surface_forms": ["kubernetes", "k8s"]},
    "ci/cd":         {"category": "Cloud & DevOps", "surface_forms": ["ci/cd", "ci cd", "continuous integration"]},
    "terraform":     {"category": "Cloud & DevOps", "surface_forms": ["terraform"]},
    "jenkins":       {"category": "Cloud & DevOps", "surface_forms": ["jenkins"]},
    "ansible":       {"category": "Cloud & DevOps", "surface_forms": ["ansible"]},
    "linux":         {"category": "Cloud & DevOps", "surface_forms": ["linux"]},

    # ---------------- Tools & Platforms ----------------
    "git":           {"category": "Tools & Platforms", "surface_forms": ["git","Git"]},
    "jira":          {"category": "Tools & Platforms", "surface_forms": ["jira"]},
    "power bi":      {"category": "Tools & Platforms", "surface_forms": ["power bi", "powerbi","Power BI"]},
    "tableau":       {"category": "Tools & Platforms", "surface_forms": ["tableau","Tableau"]},
    "excel":         {"category": "Tools & Platforms", "surface_forms": ["excel","Advanced MS Exel","Microsoft Excel"]},
    "figma":         {"category": "Tools & Platforms", "surface_forms": ["figma"]},
    "postman":       {"category": "Tools & Platforms", "surface_forms": ["postman"]},
    "spark":         {"category": "Tools & Platforms", "surface_forms": ["apache spark", "spark"]},
    "hadoop":        {"category": "Tools & Platforms", "surface_forms": ["hadoop"]},
    "airflow":       {"category": "Tools & Platforms", "surface_forms": ["airflow"]},
    "machine learning": {"category": "Tools & Platforms", "surface_forms": ["machine learning","Machine Learning"]},
    "deep learning":    {"category": "Tools & Platforms", "surface_forms": ["deep learning","Deep Learning"]},
    "nlp":              {"category": "Tools & Platforms", "surface_forms": ["nlp", "natural language processing","NLP","Natural Language Processing"]},
    "data analysis":    {"category": "Tools & Platforms", "surface_forms": ["data analysis","Exploratory Data Analysis","EDA"]},
    "rest api":         {"category": "Tools & Platforms", "surface_forms": ["rest api", "restful api"]},

    # ---------------- Soft Skills ----------------
    "communication skills": {"category": "Soft Skills", "surface_forms": ["communication skills", "communication","Communication"]},
    "leadership":           {"category": "Soft Skills", "surface_forms": ["leadership","Leadership"]},
    "teamwork":             {"category": "Soft Skills", "surface_forms": ["teamwork", "team player","Team work"]},
    "mentoring":            {"category": "Soft Skills", "surface_forms": ["mentoring", "mentorship"]},
    "problem solving":      {"category": "Soft Skills", "surface_forms": ["problem solving", "problem-solving","Problem Solving"]},
    "time management":      {"category": "Soft Skills", "surface_forms": ["time management"]},
    "collaboration":        {"category": "Soft Skills", "surface_forms": ["collaboration", "cross-functional"]},
    "adaptability":         {"category": "Soft Skills", "surface_forms": ["adaptability", "adaptable","Adaptability"]},
    "critical thinking":    {"category": "Soft Skills", "surface_forms": ["critical thinking"]},
}

# Backwards-compatible flat set (some callers just want "is this a soft skill?")
SOFT_SKILLS: Set[str] = {k for k, v in SKILLS_DB.items() if v["category"] == "Soft Skills"}

# Aliases / abbreviations used for near-match ("semantic-lite") skill matching
# when a resume uses different wording than the JD for the same underlying
# skill (e.g. JD says "ML", resume says "Machine Learning").
SKILL_SYNONYMS: Dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "sklearn": "scikit-learn",
    "nodejs": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "vuejs": "vue",
    "restful api": "rest api",
    "api development": "rest api",
    "oop": "python",  # weak signal, kept intentionally narrow
    "ci cd": "ci/cd",
    "continuous integration": "ci/cd",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
}


def get_category(skill: str) -> str:
    return SKILLS_DB.get(skill, {}).get("category", "Other")


def all_categories() -> List[str]:
    return CATEGORIES


# ---------------------------------------------------------------------------
# Domain families -- used only to give a lightweight "your projects lean
# toward X, this JD leans toward Y" nudge. Deliberately coarse: this is a
# suggestion signal, not a taxonomy of its own.
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "NLP": ["nlp", "natural language processing", "spacy", "nltk", "text classification",
            "sentiment analysis", "chatbot", "llm", "large language model","LLM","NLP""transformer",
            "named entity recognition", "text mining"],
    "computer vision": ["computer vision", "opencv", "image classification", "object detection",
                         "cnn","CNN","RNN","Image Classification","Object Detection","image processing", "image segmentation", "facial recognition"],
    "data engineering": ["airflow", "spark", "hadoop", "etl", "data pipeline", "data pipelines",
                          "kafka", "data warehouse", "data lake"],
    "web development": ["react", "django", "flask", "node.js", "express", "rest api",
                         "frontend", "front-end", "backend", "back-end", "full stack", "full-stack"],
    "cloud & devops": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "terraform",
                        "infrastructure as code", "site reliability"],
    "mobile": ["android", "ios", "swift", "kotlin", "react native", "flutter"],
}

# Short, actionable nudges for common JD-required skills a resume is missing.
# Not exhaustive by design -- anything absent here falls back to a generic
# but still concrete suggestion built at call time.
SKILL_LEARNING_TIPS: Dict[str, str] = {
    "tensorflow": "Learn TensorFlow for deep learning workflows -- even a small trained-and-evaluated model strengthens this.",
    "pytorch": "Get hands-on with PyTorch -- a short training/evaluation notebook is enough to speak to it credibly.",
    "docker": "Add Docker if you've containerized any project -- even a simple Dockerfile for a personal project counts.",
    "kubernetes": "Kubernetes is a bigger lift -- consider a basic deployment walkthrough (e.g. minikube) if you have time before applying.",
    "ci/cd": "Mention any CI/CD pipeline you've set up or used (GitHub Actions, Jenkins, GitLab CI) -- this is often assumed knowledge, not asked outright.",
    "aws": "Highlight any AWS usage, even free-tier personal projects (S3, EC2, Lambda) -- specific services matter more than 'AWS experience' as a phrase.",
    "azure": "Highlight any Azure usage -- specific services (App Service, Functions, Blob Storage) read stronger than the umbrella term.",
    "gcp": "Highlight any GCP usage -- specific services (Cloud Run, BigQuery, Vertex AI) read stronger than the umbrella term.",
    "kubernetes": "Consider a basic Kubernetes walkthrough (e.g. minikube) if you have time before applying.",
    "spark": "Mention any large-scale data processing work, even a course project using PySpark.",
    "airflow": "If you've scheduled or orchestrated any data jobs, mention it -- Airflow specifically is a plus but the underlying skill (pipeline orchestration) is what matters.",
    "nlp": "Add a project involving text data -- classification, sentiment, or a chatbot all count as NLP exposure.",
    "machine learning": "Make sure at least one project explicitly walks through data prep, model training, and evaluation.",
    "deep learning": "A single project using a neural network (even a small CNN or MLP) is enough to claim this credibly.",
    "sql": "Mention specific SQL work -- writing queries, joins, or optimizing a schema -- rather than just listing SQL as a skill.",
    "rest api": "Mention any API you've built or consumed -- even a small Flask/FastAPI project demonstrates this.",
    "communication skills": "Weave communication into your bullet points (e.g. 'presented findings to stakeholders') rather than listing it as a bare skill.",
    "problem solving": "Show this through a specific example -- a bug you diagnosed, or a tricky tradeoff you navigated -- rather than the phrase itself.",
    "leadership": "If you led even a small team or initiative, name it explicitly with scope (team size, project).",
    "teamwork": "Reference a specific cross-functional or team project rather than listing 'teamwork' alone.",
}