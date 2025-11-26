# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import re

# ==========================================
# 1. 数据准备
# ==========================================
df = pd.read_csv('combined_CLEAN.csv')
df = df[df["Campus"]=="UCD"].reset_index(drop=True)

def normalize_course_id(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").upper()

df['Course_ID'] = (df['Subject_Code'].fillna('') + df['Course_Code'].fillna('').astype(str)).apply(normalize_course_id)

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
        lower_part = part.lower()
        is_or_relationship = False
        if '(' in part and ')' in part: is_or_relationship = True
        elif ' or ' in lower_part or 'one of' in lower_part: is_or_relationship = True

        if is_or_relationship and len(courses) > 1:
            structure.append(courses)
        else:
            for c in courses: structure.append([c])
    return structure

df['Prereq_Struct'] = df['Prerequisite(s)'].apply(parse_prerequisite)

# ==========================================
# 2. 构建图
# ==========================================
G = nx.DiGraph()

for _, row in df.iterrows():
    target_course = row['Course_ID']
    G.add_node(target_course, label=target_course, title=row['Title'], group='Course')

    for prereq_group in row['Prereq_Struct']:
        if not prereq_group: continue
        if len(prereq_group) == 1:
            source = prereq_group[0]
            if source == target_course: continue
            if source not in G: G.add_node(source, label=source, group='External')
            G.add_edge(source, target_course)
        else:
            or_node_id = f"OR_{target_course}_{'_'.join(prereq_group)}"
            if or_node_id not in G: G.add_node(or_node_id, label="OR", size=5, group='Logic')
            G.add_edge(or_node_id, target_course)
            for source in prereq_group:
                if source == target_course: continue
                if source not in G: G.add_node(source, label=source, group='External')
                G.add_edge(source, or_node_id)

# ==========================================
# 3. FastAPI 初始化
# ==========================================
app = FastAPI(title="UC Course Prerequisite API")

# ⚡ 跨域，允许 GitHub Pages 调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 4. 布局函数
# ==========================================
def get_optimized_tree_layout(graph, root_node):
    pos = {}
    rev_G = graph.reverse()
    layers = {}

    try:
        lengths = nx.shortest_path_length(rev_G, source=root_node)
        for node, length in lengths.items():
            if length not in layers: layers[length] = []
            layers[length].append(node)
    except nx.NetworkXNoPath: pass

    pos[root_node] = (0, 0)
    for level in sorted(layers.keys()):
        if level == 0: continue
        current_nodes = layers[level]
        node_scores = []

        for node in current_nodes:
            parents = [p for p in graph.successors(node) if p in pos]
            avg_parent_x = sum(pos[p][0] for p in parents)/len(parents) if parents else 0
            is_or_node = 1 if str(node).startswith("OR_") else 0
            node_scores.append({'node': node, 'parent_x': avg_parent_x, 'is_or': is_or_node, 'name': str(node)})

        node_scores.sort(key=lambda x: (x['parent_x'], x['is_or'], x['name']))
        sorted_nodes = [x['node'] for x in node_scores]
        layer_width = len(sorted_nodes)
        x_sep = 1.2

        for i, node in enumerate(sorted_nodes):
            x = (i - (layer_width - 1)/2) * x_sep
            y = -level * 1.5
            pos[node] = (x, y)
    return pos

# ==========================================
# 5. API 路径
# ==========================================
@app.get("/graph/{course_id}")
def get_course_graph(course_id: str):
    course_id = normalize_course_id(course_id)
    if course_id not in G:
        raise HTTPException(status_code=404, detail=f"课程 {course_id} 不存在")

    ancestors = nx.ancestors(G, course_id)
    nodes_of_interest = ancestors.union({course_id})
    sub_G = G.subgraph(nodes_of_interest)
    pos = get_optimized_tree_layout(sub_G, course_id)

    # 返回 JSON
    nodes = [
        {"id": n, "label": sub_G.nodes[n].get("label", n),
         "group": sub_G.nodes[n].get("group", "External"),
         "title": sub_G.nodes[n].get("title", ""),
         "x": pos.get(n, (0,0))[0], "y": pos.get(n, (0,0))[1]}
        for n in sub_G.nodes()
    ]
    edges = [{"source": u, "target": v} for u,v in sub_G.edges()]

    return {"nodes": nodes, "edges": edges}
