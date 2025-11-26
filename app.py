from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import networkx as nx
import re
import plotly.graph_objects as go
import json
import plotly.utils

# 初始化 Flask
app = Flask(__name__)
# 允许跨域请求 (CORS)，这样 GitHub Pages 才能访问 Render
CORS(app)

# ==========================================
# 1. 全局数据加载与图构建 (只运行一次)
# ==========================================
print("正在初始化服务器，加载数据...")

# 读取数据 (确保 combined_CLEAN.csv 和 app.py 在同一目录下)
try:
    df = pd.read_csv('combined_CLEAN.csv')
    df = df[df["Campus"]=="UCD"].reset_index(drop=True)
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame() # 空防止崩溃

def normalize_course_id(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").upper()

if not df.empty:
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

if not df.empty:
    df['Prereq_Struct'] = df['Prerequisite(s)'].apply(parse_prerequisite)

# 构建图
G = nx.DiGraph()
if not df.empty:
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

print(f"初始化完成。节点数: {len(G.nodes)}")

# ==========================================
# 2. 布局算法 (保持不变)
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
            node_scores.append({
                'node': node,
                'parent_x': avg_parent_x, 
                'is_or': is_or_node,      
                'name': str(node)         
            })
        node_scores.sort(key=lambda x: (x['parent_x'], x['is_or'], x['name']))
        sorted_nodes = [x['node'] for x in node_scores]
        layer_width = len(sorted_nodes)
        x_sep = 1.2 
        for i, node in enumerate(sorted_nodes):
            x = (i - (layer_width - 1) / 2) * x_sep
            y = -level * 1.5
            pos[node] = (x, y)
    return pos

# ==========================================
# 3. 绘图逻辑 (适配 API 返回)
# ==========================================
def create_plotly_json(graph_obj, plot_title, highlight_node_id):
    if len(graph_obj.nodes) == 0:
        return None

    # 计算强调集合
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

    # 计算布局
    if highlight_node_id:
        pos = get_optimized_tree_layout(graph_obj, highlight_node_id)
    else:
        pos = nx.spring_layout(graph_obj, seed=42)

    # 绘制边
    edge_x, edge_y = [], []
    for edge in graph_obj.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#ccc'), 
        hoverinfo='none',
        mode='lines'
    )

    # 绘制点
    node_x, node_y = [], []
    node_text, node_color, node_size = [], [], []
    node_ids = []

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

        # 颜色逻辑
        if highlight_node_id and node == highlight_node_id:
            node_color.append('#FFD700')  # Target: Gold
            node_size.append(40)
            node_text.append(f"📍 TARGET: {hover_str}")
        elif node in indirect_prereqs:
            node_color.append('#FF4500')  # Indirect: Red
            node_size.append(25)
            node_text.append(f"🔀 OR-Option: {hover_str}")
        elif node in direct_prereqs:
            node_color.append('#FFA500')  # Direct: Orange
            node_size.append(25)
            node_text.append(f"⚡ Direct Prereq: {hover_str}")
        else:
            if group == 'Logic':
                node_color.append('#ff6b6b')
                node_size.append(8)
                node_text.append("Logic (OR)")
            elif group == 'Course':
                node_color.append('#4dabf7')
                node_size.append(15)
                node_text.append(hover_str)
            else:
                node_color.append('#adb5bd')
                node_size.append(12)
                node_text.append(f"External: {label}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers', # 只保留 Markers
        hoverinfo='text',
        hovertext=node_text,
        customdata=node_ids, # 关键：传回 ID 供前端使用
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=1.5,
            line_color='white'
        )
    )

# ... (前面的代码保持不变)

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    # 🔴 修改点 1: title 变成字典结构，不再使用 titlefont_size
                    title={
                        'text': f'<br>Prerequisite Tree: {highlight_node_id}',
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 16} # 字体大小放在这里
                    },
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    clickmode='event+select'
                )
    )
    
    # 转换为 JSON 格式返回
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

# ==========================================
# 4. API 路由接口
# ==========================================

@app.route('/')
def home():
    return "UCD Course Graph API is Running!"

@app.route('/api/graph/<course_id>', methods=['GET'])
def get_course_graph(course_id):
    """
    API 端点：接收 course_id，返回 Plotly 图表 JSON
    """
    target = normalize_course_id(course_id)
    
    if target not in G:
        return jsonify({"error": f"Course {target} not found"}), 404
        
    # 计算子图
    try:
        ancestors = nx.ancestors(G, target)
        nodes_of_interest = ancestors.union({target})
        sub_G = G.subgraph(nodes_of_interest)
        
        # 生成图表数据
        graph_json = create_plotly_json(sub_G, f"Prerequisite Tree: {target}", target)
        return jsonify(graph_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 注意：在 Render 上部署时，不要使用 debug=True
    app.run(host='0.0.0.0', port=5000)
