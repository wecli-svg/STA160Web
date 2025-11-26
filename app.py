import pandas as pd
import networkx as nx
import re
import plotly.graph_objects as go
import streamlit as st

# ================================
# 1. 载入数据
# ================================
st.set_page_config(layout="wide")
st.title("UC Davis Course Prerequisite Explorer (Network Version)")

@st.cache_data
def load_data():
    df = pd.read_csv("combined_CLEAN.csv")
    df = df[df["Campus"] == "UCD"].reset_index(drop=True)
    return df

df = load_data()

def normalize_course_id(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").upper()

df['Course_ID'] = (df['Subject_Code'].fillna('') + df['Course_Code'].fillna('').astype(str)).apply(normalize_course_id)


# ================================
# 2. prerequisite parser
# ================================
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
        if '(' in part and ')' in part:
            is_or_relationship = True
        elif ' or ' in lower_part or 'one of' in lower_part:
            is_or_relationship = True
        if is_or_relationship and len(courses) > 1:
            structure.append(courses)
        else:
            for c in courses:
                structure.append([c])
    return structure

df['Prereq_Struct'] = df['Prerequisite(s)'].apply(parse_prerequisite)


# ================================
# 3. 构建图
# ================================
@st.cache_data
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        target_course = row['Course_ID']
        G.add_node(target_course, label=target_course,
                   title=row['Title'], group='Course')

        for prereq_group in row['Prereq_Struct']:
            if not prereq_group:
                continue
            if len(prereq_group) == 1:
                source = prereq_group[0]
                if source not in G:
                    G.add_node(source, label=source, group='External')
                G.add_edge(source, target_course)
            else:
                or_node_id = f"OR_{target_course}_{'_'.join(prereq_group)}"
                if or_node_id not in G:
                    G.add_node(or_node_id, label="OR", size=5, group="Logic")
                G.add_edge(or_node_id, target_course)
                for s in prereq_group:
                    if s not in G:
                        G.add_node(s, label=s, group='External')
                    G.add_edge(s, or_node_id)
    return G

G = build_graph(df)


# ================================
# 4. 布局
# ================================
def get_optimized_tree_layout(graph, root_node):
    pos = {}
    rev_G = graph.reverse()
    layers = {}

    try:
        lengths = nx.shortest_path_length(rev_G, source=root_node)
        for node, length in lengths.items():
            if length not in layers:
                layers[length] = []
            layers[length].append(node)
    except nx.NetworkXNoPath:
        pass

    pos[root_node] = (0, 0)

    for level in sorted(layers.keys()):
        if level == 0:
            continue

        current_nodes = layers[level]
        node_scores = []

        for node in current_nodes:
            parents = [p for p in graph.successors(node) if p in pos]
            if parents:
                avg_parent_x = sum(pos[p][0] for p in parents) / len(parents)
            else:
                avg_parent_x = 0

            is_or = 1 if str(node).startswith("OR_") else 0

            node_scores.append({
                'node': node,
                'parent_x': avg_parent_x,
                'is_or': is_or,
                'name': str(node),
            })

        node_scores.sort(key=lambda x: (x['parent_x'], x['is_or'], x['name']))
        sorted_nodes = [x['node'] for x in node_scores]

        x_sep = 1.2
        for i, node in enumerate(sorted_nodes):
            x = (i - (len(sorted_nodes) - 1) / 2) * x_sep
            y = -level * 1.5
            pos[node] = (x, y)

    return pos


# ================================
# 5. Plotly 可视化
# ================================
def visualize_network_plotly(graph_obj, highlight_node=None):
    if len(graph_obj.nodes) == 0:
        return go.Figure()

    direct_prereqs = set()
    indirect_prereqs = set()

    if highlight_node and highlight_node in graph_obj:
        parents = list(graph_obj.predecessors(highlight_node))
        for p in parents:
            if graph_obj.nodes[p].get('group') == "Logic":
                indirect_prereqs.update(graph_obj.predecessors(p))
            else:
                direct_prereqs.add(p)

    if highlight_node:
        pos = get_optimized_tree_layout(graph_obj, highlight_node)
    else:
        pos = nx.spring_layout(graph_obj, seed=42)

    # edges
    edge_x, edge_y = [], []
    for u, v in graph_obj.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#ccc", width=0.8),
        hoverinfo="none"
    )

    # nodes
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for n in graph_obj.nodes():
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        group = graph_obj.nodes[n].get("group", "External")
        title = graph_obj.nodes[n].get("title", "")
        label = graph_obj.nodes[n].get("label", n)

        if n == highlight_node:
            node_color.append("#FFD700")
            node_size.append(40)
            hover_str = f"<b>{label}</b><br>{title}<br>(Target)"
        elif n in indirect_prereqs:
            node_color.append("#FF4500")
            node_size.append(25)
            hover_str = f"<b>{label}</b><br>{title}<br>(OR Option)"
        elif n in direct_prereqs:
            node_color.append("#FFA500")
            node_size.append(25)
            hover_str = f"<b>{label}</b><br>{title}<br>(Direct Prereq)"
        else:
            if group == "Logic":
                node_color.append("#ff6b6b")
                node_size.append(10)
                hover_str = "Logic Node (OR)"
            elif group == "Course":
                node_color.append("#4dabf7")
                node_size.append(15)
                hover_str = f"<b>{label}</b><br>{title}"
            else:
                node_color.append("#adb5bd")
                node_size.append(12)
                hover_str = f"External: {label}"

        node_text.append(hover_str)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="white")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Prerequisite Graph for {highlight_node}",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        hovermode="closest",
        height=800
    )
    return fig


# ================================
# 6. Streamlit 交互界面
# ================================
st.sidebar.header("Options")

course_input = st.sidebar.text_input("输入课程 ID:", "EAE126")
course_input = normalize_course_id(course_input)

if st.sidebar.button("生成图"):
    if course_input not in G:
        st.error(f"课程 {course_input} 不存在")
    else:
        ancestors = nx.ancestors(G, course_input)
        sub_G = G.subgraph(ancestors.union({course_input}))

        fig = visualize_network_plotly(sub_G, highlight_node=course_input)
        st.plotly_chart(fig, use_container_width=True)
