from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import json

app = FastAPI()

# 允许 GitHub Pages 请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 加载课程数据 & 构建图
# -------------------------
df = pd.read_csv("combined_CLEAN.csv")
df = df[df["Campus"]=="UCD"].reset_index(drop=True)

def normalize_course_id(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").upper()

df['Course_ID'] = (df['Subject_Code'].fillna('') + df['Course_Code'].fillna('').astype(str)).apply(normalize_course_id)

import re
def parse_prerequisite(prereq_text):
    if not isinstance(prereq_text, str) or not prereq_text.strip():
        return []
    clean_text = prereq_text.replace("\xa0", " ")
    and_parts = [p.strip() for p in clean_text.split(";")]
    structure = []
    for part in and_parts:
        raw_courses = re.findall(r'\b[A-Z&]{2,5}\s*\d+[A-Z]*\b', part)
        if not raw_courses: continue
        courses = [normalize_course_id(c) for c in raw_courses]
        if (" or " in part.lower()) and len(courses) > 1:
            structure.append(courses)
        else:
            for c in courses: structure.append([c])
    return structure

df['Prereq_Struct'] = df['Prerequisite(s)'].apply(parse_prerequisite)

# 构建图
G = nx.DiGraph()
for _, row in df.iterrows():
    target = row["Course_ID"]
    G.add_node(target, title=row["Title"])
    for group in row["Prereq_Struct"]:
        if len(group) == 1:
            G.add_edge(group[0], target)
        else:
            or_id = f"OR_{target}_{'_'.join(group)}"
            G.add_node(or_id, title="OR")
            G.add_edge(or_id, target)
            for c in group:
                G.add_edge(c, or_id)


# -------------------------
# API: 返回某课程的先修子图
# -------------------------
@app.get("/subgraph/{course_id}")
def get_prereq_subgraph(course_id: str):
    course_id = normalize_course_id(course_id)
    if course_id not in G:
        return {"error": "Course not found"}

    nodes = nx.ancestors(G, course_id).union({course_id})
    subG = nx.subgraph(G, nodes)
    data = nx.node_link_data(subG)
    return data
