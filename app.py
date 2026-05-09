"""
NeuroForge - Autonomous Software Engineering Agent
AMD Developer Hackathon 2026
"""
import gradio as gr
import requests
import json
import time
import tempfile
import os

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"


def call_qwen(messages, temperature=0.3):
    """Call Qwen model via vLLM."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False
    }
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def call_qwen_stream(messages, temperature=0.3):
    """Call Qwen model via vLLM with streaming."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": True
    }
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
    except Exception as e:
        yield f"Error: {str(e)}"


def plan_task(requirement, tech_stack):
    """Step 1: Create implementation plan."""
    system_msg = "You are an expert software architect. Break down the task into 3-5 clear implementation steps. Return ONLY a numbered list."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Tech Stack: {tech_stack}\n\nRequirement: {requirement}\n\nCreate implementation plan:"}
    ]
    return call_qwen(messages, 0.2)


def generate_code(step, context):
    """Step 2: Generate code for a step."""
    system_msg = "You are a senior software engineer. Write clean, well-commented Python code with error handling."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context: {context}\n\nTask: {step}\n\nWrite the code:"}
    ]
    return call_qwen(messages, 0.2)


def review_code(code, requirements):
    """Step 3: Review code quality."""
    system_msg = "You are a code reviewer. Check for bugs, security issues, and improvements. Return a brief review."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Requirements: {requirements}\n\nCode:\n{code}\n\nReview:"}
    ]
    return call_qwen(messages, 0.3)


def save_code_to_file(code: str) -> str:
    """Save generated code to a temp file and return path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="neuroforge_"
    )
    tmp.write(code)
    tmp.close()
    return tmp.name


# CSS for syntax highlighting + UI polish
CUSTOM_CSS = """
/* ── Base ── */
body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ── Header ── */
.neuroforge-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 1.5rem;
    text-align: center;
}

/* ── Buttons ── */
.gr-button-primary {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: 1px solid #2ea043 !important;
    color: white !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    box-shadow: 0 0 12px rgba(46, 160, 67, 0.4) !important;
}

/* ── Download button ── */
.download-btn {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: 1px solid #388bfd !important;
    color: white !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.download-btn:hover {
    box-shadow: 0 0 12px rgba(56, 139, 253, 0.4) !important;
}

/* ── Code blocks ── */
.code-output pre, .code-output code {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
}

/* Keyword highlighting via CSS (Gradio renders markdown code blocks) */
.code-output .keyword   { color: #ff7b72; }
.code-output .string    { color: #a5d6ff; }
.code-output .comment   { color: #8b949e; font-style: italic; }
.code-output .function  { color: #d2a8ff; }
.code-output .number    { color: #79c0ff; }

/* ── Status log ── */
.status-log {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    font-size: 0.82rem !important;
    color: #8b949e !important;
}

/* ── Inputs ── */
textarea, .gr-input, .gr-dropdown {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs ── */
.tab-nav button {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
}
.tab-nav button.selected {
    background: #21262d !important;
    color: #e6edf3 !important;
    border-bottom-color: #238636 !important;
}
"""

HIGHLIGHT_JS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script>
  // Re-run highlight.js whenever Gradio updates the DOM
  const observer = new MutationObserver(() => {
      document.querySelectorAll('pre code:not(.hljs)').forEach(block => {
          hljs.highlightElement(block);
      });
  });
  observer.observe(document.body, { childList: true, subtree: true });
</script>
"""


def process_request(requirement, tech_stack):
    """Main processing pipeline with real-time streaming updates."""
    log_lines = []
    generated_code = ""

    def log(icon, phase, msg):
        log_lines.append(f"{icon} **{phase}**: {msg}")
        return "\n\n".join(log_lines)

    # ── Step 1: Plan ──────────────────────────────────────────
    status = log("🔍", "Planning", "Analyzing requirements...")
    yield status, "", None

    plan = plan_task(requirement, tech_stack)
    steps = [s for s in plan.split("\n") if s.strip() and s[0].isdigit()]

    status = log("📋", "Plan", f"{len(steps)} steps identified")
    yield status, f"```\n{plan}\n```", None

    # ── Step 2: Code (streamed per step) ──────────────────────
    all_code_blocks = []

    for i, step in enumerate(steps[:3], 1):
        status = log("⚙️", f"Step {i}/{min(len(steps),3)}", step.strip())
        yield status, f"```\n{plan}\n```", None

        # stream the code for this step
        system_msg = "You are a senior software engineer. Write clean, well-commented Python code with error handling."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context: Goal: {requirement}\n\nTask: {step}\n\nWrite the code:"}
        ]

        streamed = f"# Step {i}: {step.strip()}\n"
        for token in call_qwen_stream(messages, 0.2):
            streamed += token
            all_code = "\n\n".join(all_code_blocks) + "\n\n" + streamed
            yield status, f"```python\n{all_code}\n```", None

        all_code_blocks.append(streamed)
        status = log("✅", f"Step {i}", "Complete")
        yield status, f"```python\n{chr(10).join(all_code_blocks)}\n```", None

    generated_code = "\n\n".join(all_code_blocks)

    # ── Step 3: Review ────────────────────────────────────────
    status = log("🔎", "Review", "Checking code quality...")
    yield status, f"```python\n{generated_code}\n```", None

    review = review_code(generated_code, requirement)
    status = log("🎉", "Done", "All steps complete!")
    yield status, f"```python\n{generated_code}\n```", None

    # ── Final output with download ────────────────────────────
    final_md = f"```python\n{generated_code}\n```\n\n---\n\n### 🔍 Code Review\n\n{review}"
    tmp_path = save_code_to_file(generated_code)
    yield status, final_md, tmp_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="NeuroForge - AI Software Engineer",
    css=CUSTOM_CSS,
    head=HIGHLIGHT_JS,
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    gr.Markdown(
        """# 🧠 NeuroForge
### Autonomous Software Engineering Agent
**Powered by AMD MI300X + Qwen2.5-Coder-14B**
"""
    )

    with gr.Row():
        # ── Left panel ──────────────────────────────────────
        with gr.Column(scale=1):
            req_input = gr.Textbox(
                label="What do you want to build?",
                lines=6,
                placeholder="e.g. A REST API for user authentication with JWT tokens...",
            )
            stack_input = gr.Dropdown(
                label="Tech Stack",
                choices=[
                    "Python/FastAPI",
                    "Python/Flask",
                    "Python/Django",
                    "Node.js/Express",
                    "Python/Streamlit",
                ],
                value="Python/FastAPI",
            )
            btn = gr.Button("⚡ Generate Software", variant="primary", size="lg")

            gr.Markdown("---")

            # Agent status log
            status_out = gr.Markdown(
                label="Agent Log",
                value="*Waiting for task...*",
                elem_classes=["status-log"],
            )

        # ── Right panel ─────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📄 Generated Code"):
                    code_out = gr.Markdown(
                        value="*Code will appear here as it is generated...*",
                        elem_classes=["code-output"],
                    )
                    download_btn = gr.File(
                        label="⬇️ Download Generated Code",
                        visible=False,
                        elem_classes=["download-btn"],
                    )

    # ── Wire up ───────────────────────────────────────────────
    def run(req, stack):
        for status, code, tmp in process_request(req, stack):
            if tmp:
                yield status, code, gr.update(value=tmp, visible=True)
            else:
                yield status, code, gr.update(visible=False)

    btn.click(
        fn=run,
        inputs=[req_input, stack_input],
        outputs=[status_out, code_out, download_btn],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
