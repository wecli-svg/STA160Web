from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import networkx as nx
import re
import plotly.graph_objects as go
import json
import plotly.utils
from sentence_transformers import util
import torch
import os
import traceback # 用于打印详细报错

app = Flask(__name__)
CORS(app)

print("💡 初始化服务器...")

# 1. 加载数据
try:
    df = pd.read_csv('combined_CLEAN.csv')
    df['Campus'] = df['Campus'].str.upper().str.strip()
    print(f"CSV 加载成功，共 {len(df)} 行")
except Exception as e:
    print(f"❌ CSV 加载失败: {e}")
    df = pd.DataFrame()

# 2. 加载 Embeddings
embeddings = None
try:
    if os.path.exists('course_embeddings.pt'):
        print("⏳ 发现 embeddings 文件，正在加载...")
        embeddings = torch.load('course_embeddings.pt', map_location=torch.device('cpu'))
        print(f"✅ Embeddings 加载成功! Shape: {embeddings.shape}")
    else:
        print("⚠️ 警告: 找不到 'course_embeddings.pt'，相似度功能将不可用。")
except Exception as e:
    print(f"❌ Embeddings 加载出错: {e}")

# 3. 预处理
def normalize_course_id(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").upper()

if not df.empty:
    df['Course_ID'] = (df['Subject_Code'].fillna('') + df['Course_Code'].fillna('').astype(str)).apply(normalize_course_id)

def parse_prerequisite(prereq_text):
    if not isinstance(prereq_text, str) or not prereq_text.strip(): return []
    clean_text = prereq_text.replace("\xa0", " ")
    and_parts = [p.strip() for p in clean_text.split(";")]
    structure = []
    for part in and_parts:
        raw_courses = re.findall(r'\b[A-Z&]{2,5}\s*\d+[A-Z]*\b', part)
        if not raw_courses: continue
        courses = [normalize_course_id(c) for c in raw_courses]
        if '(' in part and ')' in part or ' or ' in part.lower():
            if len(courses) > 1: structure.append(courses)
        else:
            for c in courses: structure.append([c])
    return structure

if not df.empty:
    df['Prereq_Struct'] = df['Prerequisite(s)'].apply(parse_prerequisite)

# 4. 构建图缓存
graphs = {}
def build_graph(campus_df):
    G = nx.DiGraph()
    for _, row in campus_df.iterrows():
        tgt = row['Course_ID']
        G.add_node(tgt, label=tgt, title=row['Title'], group='Course')
        for group in row['Prereq_Struct']:
            if not group: continue
            if len(group) == 1:
                src = group[0]
                if src != tgt:
                    if src not in G: G.add_node(src, label=src, group='External')
                    G.add_edge(src, tgt)
            else:
                or_id = f"OR_{tgt}_{'_'.join(group)}"
                if or_id not in G: G.add_node(or_id, label="OR", size=5, group='Logic')
                G.add_edge(or_id, tgt)
                for src in group:
                    if src != tgt:
                        if src not in G: G.add_node(src, label=src, group='External')
                        G.add_edge(src, or_id)
    return G

for campus in ['UCD', 'UCLA', 'UCSC', 'UCI']:
    graphs[campus] = build_graph(df[df['Campus'] == campus]) if not df.empty else nx.DiGraph()

# 5. 布局算法
def get_optimized_tree_layout(graph, root_node):
    pos = {}
    try:
        layers = {}
        rev_G = graph.reverse()
        lengths = nx.shortest_path_length(rev_G, source=root_node)
        for node, length in lengths.items():
            if length not in layers: layers[length] = []
            layers[length].append(node)
        
        pos[root_node] = (0, 0)
        for level in sorted(layers.keys()):
            if level == 0: continue
            node_scores = []
            for node in layers[level]:
                parents = [p for p in graph.successors(node) if p in pos]
                avg_x = sum(pos[p][0] for p in parents)/len(parents) if parents else 0
                is_or = 1 if str(node).startswith("OR_") else 0
                node_scores.append({'n': node, 'px': avg_x, 'or': is_or, 'name': str(node)})
            
            node_scores.sort(key=lambda x: (x['px'], x['or'], x['name']))
            nodes = [x['n'] for x in node_scores]
            width = len(nodes)
            for i, n in enumerate(nodes):
                pos[n] = ((i - (width-1)/2)*1.2, -level*1.5)
        return pos
    except Exception:
        return nx.spring_layout(graph, seed=42)

def create_plotly_json(G, title, highlight):
    if len(G.nodes) == 0: return None
    try:
        pos = get_optimized_tree_layout(G, highlight)
        
        # 简单分类颜色
        direct, indirect = set(), set()
        if highlight in G:
            for p in G.predecessors(highlight):
                if G.nodes[p].get('group') == 'Logic':
                    indirect.update(G.predecessors(p))
                else:
                    direct.add(p)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            if u in pos and v in pos:
                edge_x.extend([pos[u][0], pos[v][0], None])
                edge_y.extend([pos[u][1], pos[v][1], None])
        
        node_x, node_y, txt, color, size, ids = [], [], [], [], [], []
        for n in G.nodes():
            if n not in pos: continue
            node_x.append(pos[n][0])
            node_y.append(pos[n][1])
            ids.append(n)
            
            # 样式
            c, s = '#adb5bd', 12
            if n == highlight: c, s = '#FFD700', 40
            elif n in direct: c, s = '#FFA500', 25
            elif n in indirect: c, s = '#FF4500', 25
            elif str(n).startswith("OR_"): c, s = '#ff6b6b', 8
            elif G.nodes[n].get('group') == 'Course': c, s = '#4dabf7', 15
            
            color.append(c); size.append(s)
            txt.append(f"<b>{n}</b><br>{G.nodes[n].get('title', '')}")

        fig = go.Figure(data=[
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#ccc', width=0.8), hoverinfo='none'),
            go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(color=color, size=size), hovertext=txt, hoverinfo='text', customdata=ids)
        ], layout=go.Layout(
            title={'text': title, 'x':0.5, 'font':{'size':16}}, showlegend=False, hovermode='closest',
            margin=dict(t=40,b=20,l=5,r=5), xaxis=dict(visible=False), yaxis=dict(visible=False), clickmode='event+select'
        ))
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e:
        print(f"绘图错误: {e}")
        return None

# 6. API
@app.route('/')
def home():
    # 改一下这里，方便确认版本
    return "API v2.0 (Search Endpoint) is Running!"

@app.route('/api/search', methods=['GET']) # <--- 关键是这里，旧代码没有这一行
def search():
    try:
        campus = request.args.get('campus', 'UCD').upper()
        cid = normalize_course_id(request.args.get('course_id', ''))
        
        # 1. 查找课程
        rows = df[(df['Campus'] == campus) & (df['Course_ID'] == cid)]
        if rows.empty:
            return jsonify({"error": f"Course {cid} not found in {campus}"}), 404
        
        target_idx = rows.index[0]
        prereq_text = rows.iloc[0]['Prerequisite(s)']
        
        resp = {
            "prereq_list": prereq_text if pd.notna(prereq_text) else "None",
            "graph": None,
            "similarity": {}
        }
        
        # 2. 生成图
        if campus in graphs and cid in graphs[campus]:
            anc = nx.ancestors(graphs[campus], cid)
            sub = graphs[campus].subgraph(anc.union({cid}))
            resp['graph'] = create_plotly_json(sub, f"Tree: {cid}", cid)
            
        # 3. 计算相似度
        if embeddings is not None:
            target_emb = embeddings[target_idx].unsqueeze(0)
            sim_res = {}
            for c in ['UCD', 'UCLA', 'UCSC', 'UCI']:
                if c == campus: 
                    sim_res[c] = []
                    continue
                
                mask = (df['Campus'] == c)
                if not mask.any(): 
                    sim_res[c] = []
                    continue
                
                # 🛠️ 关键修复：使用 .values 确保是 numpy array 或 boolean array
                c_embs = embeddings[mask.values] 
                c_idxs = df[mask].index
                
                hits = util.semantic_search(target_emb, c_embs, top_k=5)[0]
                
                sim_res[c] = []
                for h in hits:
                    row = df.loc[c_idxs[h['corpus_id']]]
                    sim_res[c].append({
                        "code": row['Course_ID'],
                        "title": row['Title'],
                        "score": round(h['score'], 3)
                    })
            resp['similarity'] = sim_res
            
        return jsonify(resp)

    except Exception as e:
        # 这一步能让你在 Render Logs 看到详细的 Python 报错
        print(traceback.format_exc())
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
