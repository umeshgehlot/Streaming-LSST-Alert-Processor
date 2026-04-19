import os
import re

def create_tikz_block(title, descriptions):
    # Generates a simple TikZ flowchart block for the architectures
    # to perfectly satisfy the LaTeX compilation requirement natively.
    tikz = [
        "\\begin{tikzpicture}[",
        "  node distance=1.5cm and 2cm,",
        "  box/.style={rectangle, draw, thick, rounded corners, minimum width=3cm, minimum height=1cm, align=center, fill=blue!5},",
        "  arrow/.style={-{Stealth[scale=1.2]}, thick}",
        "]"
    ]
    
    last_node = ""
    for i, desc in enumerate(descriptions):
        node_name = f"n{i}"
        anchor = f"below=of {last_node}" if last_node else ""
        tikz.append(f"\\node[box] ({node_name}) {'' if not anchor else '[' + anchor + ']'} {{{desc}}};")
        if last_node:
            tikz.append(f"\\draw[arrow] ({last_node}) -- ({node_name});")
        last_node = node_name
        
    tikz.append("\\end{tikzpicture}")
    return "\n".join(tikz)

def main():
    tex_path = "research_paper.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Inline the \input{diagrams/...} PGFPlots
    def repl_input(match):
        filename = match.group(1)
        filepath = os.path.join("diagrams", filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as tf:
                return tf.read()
        return match.group(0)
    
    content = re.sub(r"\\input\{diagrams/([^\}]+)\}", repl_input, content)

    # 2. Replace PNG block diagrams with TikZ equivalents
    arch = create_tikz_block("System Architecture", ["Raw Astronomical Data Ingestion", "Data Preprocessing & Windowing", "Stacked Neural Ensemble Models", "Anomaly Scoring & RL Adaptation"])
    pre = create_tikz_block("Data Preprocessing", ["Raw Light Curve", "Outlier Clipping (Sigma)", "Min-Max Normalization", "Sliding Window (W=32)"])
    models = create_tikz_block("Model Architectures", ["Autoencoder (Amplitude)", "Variational Autoencoder (Latent)", "Transformer (Temporal Attention)", "Rank/Max Score Fusion Engine"])
    agent = create_tikz_block("Agent Pipeline", ["Data Collection Stage", "Feature Extraction Stage", "Model Inference Stage", "SLM Reasoning & Human Feedback"])

    content = re.sub(r"\\centerline\{\\includegraphics\[.*?\]\{system_architecture\.png\}\}", lambda m: "\\\\centerline{\n" + arch + "\n}", content)
    content = re.sub(r"\\centerline\{\\includegraphics\[.*?\]\{data_preprocessing\.png\}\}", lambda m: "\\\\centerline{\n" + pre + "\n}", content)
    content = re.sub(r"\\centerline\{\\includegraphics\[.*?\]\{model_architectures\.png\}\}", lambda m: "\\\\centerline{\n" + models + "\n}", content)
    content = re.sub(r"\\centerline\{\\includegraphics\[.*?\]\{agent_pipeline\.png\}\}", lambda m: "\\\\centerline{\n" + agent + "\n}", content)

    # 3. Handle Citations Warning - Just suppress undefined citations by providing a skeleton or ignoring. 
    # Actually the user error "Citation undefined" is just a warning in latex. We don't have to fix it,
    # but the paper will have [?]. To fix [?], we need the bibliography items.

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Inlining completed successfully!")

if __name__ == "__main__":
    main()
