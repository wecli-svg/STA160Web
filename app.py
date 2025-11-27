from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import networkx as nx
import re
import plotly.graph_objects as go
import json
import plotly.utils
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# ==========================================
# 1. 全局数据加载与处理
# ==========================================
print("💡 正在初始化服务器...")

# 1.1 读取数据
try:
    df = pd.read_csv('combined_CLEAN.csv')
    # 确保 Campus 列存在且干净
    df['Campus'] = df['Campus'].str.upper().str.strip()
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()

# 1.2 文本清洗与 Course ID 生成
def normalize_course_id(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").upper()

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    noise_phrases = ["consent of instructor", "upper division standing", "pass/no pass"]
    for phrase in noise_phrases:
        text = text.replace(phrase, "")
    return text

if not df.empty:
    df['Course_ID'] = (df['Subject_Code'].fillna('') + df['Course_Code'].fillna('').astype(str)).apply(normalize_course_id)
    
    # 准备用于 Embedding 的文本
    df["processed_text"] = (
        "Course Title: " + df["Title"].fillna("") + ". " +
        df["Title"].fillna("") + ". " +
        "Description: " + df["Course Description"].apply(clean_text)
    )

# 1.3 预解析 Prerequisite 用于图构建
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

if not df.empty:
    df['Prereq_Struct'] = df['Prerequisite(s)'].apply(parse_prerequisite)

# 1.4 模型加载与 Embedding 计算 (针对 Render 优化)
# ⚠️ 注意: Render 免费版内存有限，这里使用轻量级模型 'all-MiniLM-L6-v2'
# 如果内存依然溢出，建议在本地计算好 embeddings 保存为 .npy 文件上传读取
print("⏳ 正在加载 NLP 模型...")
device = "cpu" # 服务器通常没有 GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("⏳ 正在计算 Embedding (这可能需要几十秒)...")
if not df.empty:
    embeddings = model.encode(
        df["processed_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
else:
    embeddings = None

print("✅ 初始化完成。")

# ==========================================
# 2. 图构建逻辑 (按学校动态构建)
# ==========================================
# 为了支持多学校，我们需要根据请求动态构建图，或者缓存每个学校的图
# 这里采用动态构建子图的方式，因为只需查询特定课程的依赖

def build_prereq_graph_for_campus(campus_df):
    G = nx.DiGraph()
    for _, row in campus_df.iterrows():
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
    return G

# 缓存图对象以提升性能
graphs = {}
for campus in ['UCD', 'UCLA', 'UCSC', 'UCI']:
    campus_data = df[df['Campus'] == campus]
    if not campus_data.empty:
        graphs[campus] = build_prereq_graph_for_campus(campus_data)
    else:
        graphs[campus] = nx.DiGraph()

# ==========================================
# 3. 布局与绘图逻辑 (保持之前的优化版)
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
            if parents:
                avg_parent_x = sum(pos[p][0] for p in parents) / len(parents)
            else:
                avg_parent_x = 0
            is_or_node = 1 if str(node).startswith("OR_") else 0
            node_scores.append({'node': node, 'parent_x': avg_parent_x, 'is_or': is_or_node, 'name': str(node)})
        node_scores.sort(key=lambda x: (x['parent_x'], x['is_or'], x['name']))
        sorted_nodes = [x['node'] for x in node_scores]
        layer_width = len(sorted_nodes)
        x_sep = 1.2 
        for i, node in enumerate(sorted_nodes):
            x = (i - (layer_width - 1) / 2) * x_sep
            y = -level * 1.5
            pos[node] = (x, y)
    return pos

def create_plotly_json(graph_obj, plot_title, highlight_node_id):
    if len(graph_obj.nodes) == 0: return None

    direct_prereqs = set()
    indirect_prereqs = set()
    if highlight_node_id and highlight_node_id in graph_obj:
        predecessors = list(graph_obj.predecessors(highlight_node_id))
        for p in predecessors:
            p_group = graph_obj.nodes[p].get('group', 'External')
            if p_group == 'Logic':
                grand_parents = list(graph_obj.predecessors(p))
                indirect_prereqs.update(grand_parents)
            else:
                direct_prereqs.add(p)

    if highlight_node_id:
        pos = get_optimized_tree_layout(graph_obj, highlight_node_id)
    else:
        pos = nx.spring_layout(graph_obj, seed=42)

    edge_x, edge_y = [], []
    for edge in graph_obj.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color='#ccc'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color, node_size, node_ids = [], [], [], [], [], []
    for node in graph_obj.nodes():
        if node not in pos: continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)
        attrs = graph_obj.nodes[node]
        group = attrs.get('group', 'External')
        label = attrs.get('label', node)
        title = attrs.get('title', '')
        hover_str = f"<b>{label}</b><br>{title}"

        if highlight_node_id and node == highlight_node_id:
            node_color.append('#FFD700'); node_size.append(40); node_text.append(f"📍 TARGET: {hover_str}")
        elif node in indirect_prereqs:
            node_color.append('#FF4500'); node_size.append(25); node_text.append(f"🔀 OR-Option: {hover_str}")
        elif node in direct_prereqs:
            node_color.append('#FFA500'); node_size.append(25); node_text.append(f"⚡ Direct Prereq: {hover_str}")
        else:
            if group == 'Logic': node_color.append('#ff6b6b'); node_size.append(8); node_text.append("Logic (OR)")
            elif group == 'Course': node_color.append('#4dabf7'); node_size.append(15); node_text.append(hover_str)
            else: node_color.append('#adb5bd'); node_size.append(12); node_text.append(f"External: {label}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_text, customdata=node_ids,
        marker=dict(showscale=False, color=node_color, size=node_size, line_width=1.5, line_color='white')
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title={'text': f'<br>Prerequisite Tree: {highlight_node_id}', 'x':0.5, 'font': {'size': 16}},
                    showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white', clickmode='event+select'
                ))
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

# ==========================================
# 4. API 路由接口 (核心业务逻辑)
# ==========================================

@app.route('/')
def home():
    return "UC Course API is Running!"

@app.route('/api/search', methods=['GET'])
def search_course():
    """
    参数:
        campus: 学校代码 (UCD, UCLA, UCSC, UCI)
        course_id: 课程代码 (e.g. EAE126)
    返回:
        JSON: {
            "graph": Plotly JSON,
            "prereq_list": "String description",
            "similarity": { "UCD": [], "UCLA": [], ... }
        }
    """
    campus = request.args.get('campus', 'UCD').upper()
    course_id = normalize_course_id(request.args.get('course_id', ''))
    
    if not course_id:
        return jsonify({"error": "Please provide a Course ID"}), 400

    # 1. 在指定 Campus 找到目标课程
    # 使用原始 df 查找行
    target_rows = df[(df['Campus'] == campus) & (df['Course_ID'] == course_id)]
    
    if target_rows.empty:
        return jsonify({"error": f"Course {course_id} not found in {campus}"}), 404
    
    target_row = target_rows.iloc[0]
    target_idx = target_rows.index[0] # 全局索引
    
    response_data = {}

    # --- 2. 获取 Prerequisite List (Raw Text) ---
    raw_prereq = target_row['Prerequisite(s)']
    response_data['prereq_list'] = raw_prereq if pd.notna(raw_prereq) else "No prerequisites listed."

    # --- 3. 生成 Graph (可视化) ---
    G_campus = graphs.get(campus)
    if G_campus and course_id in G_campus:
        ancestors = nx.ancestors(G_campus, course_id)
        nodes_of_interest = ancestors.union({course_id})
        sub_G = G_campus.subgraph(nodes_of_interest)
        response_data['graph'] = create_plotly_json(sub_G, "", course_id)
    else:
        response_data['graph'] = None # 可能是孤立点或没找到

    # --- 4. 计算相似度 (Similarity Table) ---
    # 目标 Embedding
    target_emb = embeddings[target_idx].unsqueeze(0) # Shape (1, 384)
    
    similarity_results = {}
    target_campuses = ['UCD', 'UCLA', 'UCSC', 'UCI']
    
    for target_campus in target_campuses:
        similarity_results[target_campus] = [] # 初始化空列表
        
        # 如果是本校，不仅不计算，还要显式留空 (根据需求)
        if target_campus == campus:
            continue
            
        # 筛选目标学校的数据
        campus_mask = (df['Campus'] == target_campus)
        if not campus_mask.any(): continue
        
        campus_embeddings = embeddings[campus_mask]
        campus_indices = df[campus_mask].index
        
        # 计算相似度
        hits = util.semantic_search(target_emb, campus_embeddings, top_k=5)
        
        # 整理 Top 5
        top_hits = hits[0] # 只有1个query
        for hit in top_hits:
            global_idx = campus_indices[hit['corpus_id']] # 还原回原始 df 的 index
            similarity_results[target_campus].append({
                "code": df.loc[global_idx, 'Course_ID'],
                "title": df.loc[global_idx, 'Title'],
                "score": round(hit['score'], 3)
            })
            
    response_data['similarity'] = similarity_results

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
