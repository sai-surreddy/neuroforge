"""
NeuroForge - Autonomous Software Engineering Agent
AMD Developer Hackathon 2026
"""
import gradio as gr
import requests
import json
import tempfile
import os

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"

CUSTOM_CSS = """
/* Success popup */
.success-banner {
    background: linear-gradient(135deg, #1a4731, #166534);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 16px 24px;
    margin: 12px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: slideIn 0.4s ease;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.success-banner p {
    margin: 0;
    color: #bbf7d0;
    font-size: 1rem;
    font-weight: 600;
}

/* Download button glow */
.download-row {
    animation: fadeIn 0.5s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.97); }
    to   { opacity: 1; transform: scale(1); }
}
.download-row .gr-button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: 1px solid #3b82f6 !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    border-radius: 8px !important;
    box-shadow: 0 0 18px rgba(59,130,246,0.5) !important;
    transition: box-shadow 0.2s ease !important;
}
.download-row .gr-button:hover {
    box-shadow: 0 0 28px rgba(59,130,246,0.8) !important;
}
"""


def call_qwen(messages, temperature=0.3):
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
    system_msg = "You are an expert software architect. Break down the task into 3-5 clear implementation steps. Return ONLY a numbered list."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Tech Stack: {tech_stack}\n\nRequirement: {requirement}\n\nCreate implementation plan:"}
    ]
    return call_qwen(messages, 0.2)


def review_code(code, requirements):
    system_msg = "You are a code reviewer. Check for bugs, security issues, and improvements. Return a brief review."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Requirements: {requirements}\n\nCode:\n{code}\n\nReview:"}
    ]
    return call_qwen(messages, 0.3)


def save_code_to_file(code):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="neuroforge_"
    )
    tmp.write(code)
    tmp.close()
    return tmp.name


def process_request(requirement, tech_stack):
    log_lines = []
    all_code_blocks = []

    def log(icon, phase, msg):
        log_lines.append(f"{icon} **{phase}**: {msg}")
        return "\n\n".join(log_lines)

    # Step 1: Plan
    status = log("🔍", "Planning", "Analyzing requirements...")
    yield status, "*Generating plan...*", gr.update(visible=False), gr.update(visible=False)

    plan = plan_task(requirement, tech_stack)
    steps = [s for s in plan.split("\n") if s.strip() and s[0].isdigit()]

    status = log("📋", "Plan", f"{len(steps)} steps identified")
    yield status, f"**Plan:**\n```\n{plan}\n```", gr.update(visible=False), gr.update(visible=False)

    # Step 2: Code generation
    for i, step in enumerate(steps[:3], 1):
        status = log("⚙️", f"Step {i}/{min(len(steps), 3)}", step.strip())
        yield status, f"**Plan:**\n```\n{plan}\n```\n\n*Generating step {i}...*", gr.update(visible=False), gr.update(visible=False)

        system_msg = "You are a senior software engineer. Write clean, well-commented Python code with error handling."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context: Goal: {requirement}\n\nTask: {step}\n\nWrite the code:"}
        ]

        streamed = f"# Step {i}: {step.strip()}\n"
        for token in call_qwen_stream(messages, 0.2):
            streamed += token
            current_code = "\n\n".join(all_code_blocks) + ("\n\n" if all_code_blocks else "") + streamed
            yield status, f"**Plan:**\n```\n{plan}\n```\n\n**Generated Code:**\n```python\n{current_code}\n```", gr.update(visible=False), gr.update(visible=False)

        all_code_blocks.append(streamed)
        status = log("✅", f"Step {i}", "Complete")

    generated_code = "\n\n".join(all_code_blocks)

    # Step 3: Review
    status = log("🔎", "Review", "Checking code quality...")
    yield status, f"**Generated Code:**\n```python\n{generated_code}\n```", gr.update(visible=False), gr.update(visible=False)

    review = review_code(generated_code, requirement)
    status = log("🎉", "Done", "All steps complete!")

    tmp_path = save_code_to_file(generated_code)

    final_md = f"**Generated Code:**\n```python\n{generated_code}\n```\n\n---\n\n**Code Review:**\n\n{review}"

    # Show success banner + download button
    yield (
        status,
        final_md,
        gr.update(visible=True),           # success banner
        gr.update(value=tmp_path, visible=True),  # download file
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="NeuroForge - AI Software Engineer", css=CUSTOM_CSS) as demo:

    gr.Markdown(
        """# 🧠 NeuroForge
### Autonomous Software Engineering Agent
**Powered by AMD MI300X + Qwen2.5-Coder-14B**"""
    )

    with gr.Row():
        # Left panel
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
            gr.Markdown("### 🤖 Agent Log")
            status_out = gr.Markdown(value="*Waiting for task...*")

        # Right panel
        with gr.Column(scale=2):
            gr.Markdown("### 📄 Generated Code")
            code_out = gr.Markdown(value="*Code will appear here as it is generated...*")

            # ── Success banner (hidden until done) ──
            with gr.Row(visible=False, elem_classes=["download-row"]) as success_row:
                gr.HTML("""
                <div class="success-banner">
                    <span style="font-size:1.8rem">🎉</span>
                    <p>Code generated successfully! Click below to download.</p>
                </div>
                """)

            # ── Download file (hidden until done) ──
            download_file = gr.File(
                label="⬇️ Download Generated Code (.py)",
                visible=False,
                elem_classes=["download-row"],
            )

    def run(req, stack):
        for status, code, show_banner, show_file in process_request(req, stack):
            yield status, code, show_banner, show_file

    btn.click(
        fn=run,
        inputs=[req_input, stack_input],
        outputs=[status_out, code_out, success_row, download_file],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
